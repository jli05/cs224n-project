from __future__ import (division, absolute_import,
                        print_function, unicode_literals)
import argparse
import logging
import sys
import os
import glob
import json
import heapq
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from optimisers import adam, adadelta, rmsprop, sgd
from embeddings import gloveDocumentParser
from sklearn.cross_validation import train_test_split

EPSILON_FOR_LOG = 1e-8

def get_encoder(model):
    def baseline_encoder(x, y, x_mask, y_pos, params, tparams):
        ''' baseline context encoder given one piece of text

        Returns ctx each row for a training instance
        '''
        if x.ndim == 1:
            mb_size = 1
        elif x.ndim == 2:
            mb_size = x.shape[0]
        seq_len = params['seq_maxlen']
        wv_size = params['word_vector_size']
        
        x_emb = tparams['Xemb'][x.flatten(), :]
        x_emb_masked = T.batched_dot(x_emb, x_mask.flatten())

        if x.ndim == 1:
            ctx = x_emb_masked.sum(axis=0) / x_mask.sum()
        elif x.ndim == 2:
            ctx = T.batched_dot(
                x_emb_masked.reshape((mb_size, seq_len, wv_size)).sum(axis=1),
                1 / x_mask.sum(axis=1)
            )
         
        return ctx 

    def attention_encoder(x, y, x_mask, y_pos, params, tparams):
        ''' attention-based context encoder given one piece of text

        '''
        C = params['summary_context_length']
        Q = params['attention_weight_max_roll']
        x_embedding = tparams['Xemb'][x[:text_length], :]
        y_embedding = tparams['Yemb'][y[(summary_length - C):summary_length], :]
        p = T.nnet.softmax(
            T.dot(x_embedding, T.dot(tparams['attention_P'], y_embedding.flatten()))
        )
        # we're going to roll p with shift [-Q,Q]
        # for start and end elements in p the roll is going
        # to push them to the other end -- we know this is a problem
        p /= 2 * Q + 1
        return sum([T.dot(T.roll(p, q), x_embedding)
                    for q in range(- Q, Q + 1)]).flatten()

    if model == 'baseline':
        encoder = baseline_encoder
    elif model == 'attention':
        encoder = attention_encoder
    return encoder

def dropout_layer(state_before, params, tparams):
    trng = params['trng']
    use_noise = (params['phase'] == 'training')
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, 
                                     p=params['dropout_rate'], n=1,
                                     dtype=state_before.dtype),
        state_before * params['dropout_rate']
        )
    return proj

def conditional_distribution(x, y, x_mask, y_pos, params, tparams):
    ''' Return the conditional distribution of next summary word index

    Given the input text tensor and summary tensor, returns the distribution for the next summary word index
    '''
    encoder = get_encoder(params['model'])
    C = params['summary_context_length']
    wv_size = params['word_vector_size']

    if x.ndim == 1:
        y_emb = tparams['Yemb'][y[(y_pos - C):y_pos].flatten(), :].flatten()
        h = T.tanh(T.dot(tparams['U'], y_emb) + tparams['b'])
        ctx = encoder(x, y, x_mask, y_pos, params, tparams)

        u = T.dot(tparams['V'], h) + T.dot(tparams['W'], ctx)
        y_next = T.nnet.softmax(u)

    elif x.ndim == 2:
        mb_size = x.shape[0]
        y_emb = tparams['Yemb'][y[:, (y_pos - C):y_pos].flatten(), :]
        y_emb = y_emb.flatten().reshape((C * wv_size, mb_size))
        # each column for a training instance
        h = T.tanh(T.dot(tparams['U'], y_emb) + tparams['b'])
        # each row for a training instance
        ctx = encoder(x, y, x_mask, y_pos, params, tparams)

        # each column for a training instance
        u = T.dot(tparams['V'], h) + T.dot(tparams['W'], ctx.T)
        # softmax works row-wise
        y_next = T.nnet.softmax(u.T)

    return y_next

def conditional_score(x, y, x_mask, y_pos, params, tparams):
    ''' Return conditional score of the (j+1)-th word index of the summary i.e. y[j]

    '''
    dist = conditional_distribution(x, y, x_mask, y_pos, params, tparams)
    if x.ndim == 1:
        return dist[y[y_pos]]
    elif x.ndim == 2:
        return dist[T.arange(x.shape[0]), y[:, y_pos]]

def training_model_tensors(x, y, x_mask, y_mask, params, tparams, y_embedder):
    ''' Return tensors for training model

    '''
    mb_size = params['minibatch_size']
    C = params['summary_context_length']
    l = params['summary_maxlen']
    
    # pad y
    id_pad = y_embedder.word_to_id[y_embedder.pad]
    y_padded = T.concatenate([T.alloc(id_pad, (mb_size, C)), y], axis=1)

    # compute the model probabilities for each encoded token in y
    fn = lambda y_pos, x, y, x_mask: conditional_score(x, y, x_mask, y_pos, params, tparams)
    y_pos_range = T.arange(C, l + C, dtype='int32')

    prob_, _ = theano.scan(fn,
                           sequences=[y_pos_range],
                           outputs_info=None,
                           non_sequences=[x, y_padded, x_mask],
                           n_steps=l)
    prob = T.concatenate([v.reshape((mb_size, 1)) for v in prob_], axis=1)

    # masked negative log-likelihood
    nll_per_token = - T.log(prob + EPSILON_FOR_LOG) * y_mask
    nll_per_text = T.sum(nll_per_token, axis=1) / T.sum(y_mask, axis=1) 
    return nll_per_text

def tfunc_best_candidate_tokens(params, tparams):
    ''' Returns a Theano function that computes the best k candidate terms for the next position in the summary

    '''
    k = params['summary_search_beam_size']

    x = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    x_mask = T.cast(T.scalar(dtype=theano.config.floatX), 'int32')
    y = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    y_pos = T.cast(T.scalar(dtype=theano.config.floatX), 'int32')
    
    dist = conditional_distribution(x, y, x_mask, y_pos, params, tparams)
    best_candidate_ids = dist.argsort()[-k:] 
    f = theano.function([x, y, x_mask, y_pos],
                        [best_candidate_ids, dist[best_candidate_ids]])
    return f

def summarize(x, x_mask, f_best_candidates, params, tparams, y_embedder):
    ''' Generate summary for a single text using beam search

    Parameters
    -----------
    x : numpy vector (not Theano variable) 
        encoded single text to summarize
    x_mask : numpy vector (not Theano variable)
             mask vector for the text
    '''
    C = params['summary_context_length']
    k = params['summary_search_beam_size']
    id_pad = y_embedder.word_to_id[y_embedder.pad]

    # initialise the summary and the beams for search
    y = [y_embedder.word_to_id[y_embedder.pad]] * C
    beams = [(0.0, y)]

    for j in range(params['summary_maxlen']):
        # for each (score, y) in the current beam, expand with the 
        # k best candidates for the next position in the summary
        new_beams = []
        for (base_score, y) in beams:
            token_ids, token_probs = f_best_candidates(x, y, x_mask, len(y)) 
            for (token_id, token_prob) in zip(token_ids, token_probs):
                # add a small constant before taking log to increase 
                # numerical stability
                new_score = base_score - np.log(EPSILON_FOR_LOG + token_prob)
                heapq.heappush(new_beams, (new_score, y + [token_id]))

        # Now we retain the k best summaries after all expansions
        # for the next position
        beams = heapq.nsmallest(k, new_beams)

    (best_nll_score, summary) = heapq.heappop(beams)
    return summary[C:]

def load_params_(params, tparams, file_path):
    pass

def save_params_(params, tparams, file_path):
    pass

def init_shared_tparam_(name, shape, value=None,
                        borrow=True, dtype=theano.config.floatX):
    if value is None:
        value=np.random.uniform(low=-0.02, high=0.02, size=shape)
    return theano.shared(value=value.astype(dtype), 
                         name=name,
                         borrow=borrow)

def init_params(model,
                corpus,
                optimizer,
                learning_rate,
                summary_context_length,
                l2_penalty_coeff,
                minibatch_size,
                epochs,
                train_split,
                seed,
                dropout_rate,
                embed_full_text_by,
                internal_representation_dim,
                attention_weight_max_roll,
                generate_summary,
                summary_maxlen,
                summary_search_beam_size,
                save_params_every,
                validate_every):    
    params = {'model': model,
              'corpus': corpus,
              'optimizer': optimizer,
              'learning_rate': learning_rate,
              'summary_context_length': summary_context_length,
              'l2_penalty_coeff': l2_penalty_coeff,
              'minibatch_size': minibatch_size,
              'epochs': epochs,
              'train_split': train_split,
              'seed': seed,
              'dropout_rate': dropout_rate,
              'embed_full_text_by': embed_full_text_by,
              'internal_representation_dim': internal_representation_dim,
              'attention_weight_max_roll': attention_weight_max_roll,
              'generate_summary': generate_summary,
              'summary_maxlen': summary_maxlen,
              'summary_search_beam_size': summary_search_beam_size,
              'save_params_every': save_params_every,
              'validate_every': validate_every}
    params.update({'rng': np.random.RandomState(seed=seed),
                   'trng': RandomStreams(seed=seed)}) 
    
    if embed_full_text_by == 'word':
        embedding_x = gloveDocumentParser('glove/glove.25k.300d.txt')
        embedding_y = embedding_x
    else:
        embedding_x = None
        embedding_y = None

    tparams = {
        'U': init_shared_tparam_('U', 
                                 (internal_representation_dim, summary_context_length * embedding_y.token_dim)),
        'b': init_shared_tparam_('b',
                                 (internal_representation_dim,)),
        'V': init_shared_tparam_('V', 
                                 (embedding_y.embedding_n_tokens, internal_representation_dim)),
        'W': init_shared_tparam_('W', 
                                 (embedding_y.embedding_n_tokens, embedding_x.token_dim)),
        'Xemb': init_shared_tparam_('Xemb', embedding_x.word_to_vector_matrix.shape, 
                                    value=embedding_x.word_to_vector_matrix),
        'Yemb': init_shared_tparam_('Yemb', embedding_y.word_to_vector_matrix.shape,
                                    value=embedding_y.word_to_vector_matrix)
        }
    if model == 'attention':
        tparams.update({'attention_P': init_shared_tparam_('attention_P', 
                                                           (embedding_x.token_dim, summary_context_length * embedding_y.token_dim))})
    
    return params, tparams, embedding_x, embedding_y

def validate(params, tparams):
    pass

def load_corpus(params, tparams, embedding_x, embedding_y):
    id_pad = embedding_y.word_to_id[embedding_y.pad]
    C = params['summary_context_length']

    x_vectors = []
    x_lengths = []
    y_vectors = []
    y_lengths = []
    for file_path in glob.iglob(os.path.join(params['corpus'], '*.json')):
        try:
            with open(file_path, 'r') as f:
                document = json.load(f)
            full_text_vector = embedding_x.parseDocument(document['full_text'])
            summary_vector = embedding_y.parseDocument(document['summary'])
        
            if not len(full_text_vector) or not len(summary_vector):
                continue
            x_vectors.append(full_text_vector)
            y_vectors.append([id_pad] * C + summary_vector)
            x_lengths.append(len(full_text_vector))
            y_lengths.append(C + len(summary_vector))
        except Exception as e:
            continue
    print('Loaded {:} files'.format(len(x_vectors)))
    
    x = np.zeros((len(x_vectors), max(x_lengths)), dtype='int32')
    y = np.zeros((len(y_vectors), max(y_lengths)), dtype='int32')
    for i in range(len(x_vectors)):
        x[i, :x_lengths[i]] = x_vectors[i]
        y[i, :y_lengths[i]] = y_vectors[i]
    x_lengths = np.array(x_lengths, dtype='int32')
    y_lengths = np.array(y_lengths, dtype='int32')

    x_train, x_test, y_train, y_test, \
        x_lengths_train, x_lengths_test, \
        y_lengths_train, y_lengths_test = \
        train_test_split(x, y, x_lengths, y_lengths,
                         train_size=params['train_split'],
                         random_state=params['rng'])
    
    return x_train, x_test, y_train, y_test, \
        x_lengths_train, x_lengths_test, \
        y_lengths_train, y_lengths_test

def train(model='baseline',
          corpus=None,
          # optimiser
          optimizer='adam',
          learning_rate=0.001,
          # model params
          embed_full_text_by='word',
          seq_maxlen=500,
          summary_maxlen=500,
          summary_context_length=10,
          internal_representation_dim=1000,
          attention_weight_max_roll=5,
          # training params
          l2_penalty_coeff=0.0,
          train_split=0.75,
          epochs=float('inf'),
          minibatch_size=20,
          seed=None,
          dropout_rate=None,
          # model load/save
          save_params='ass_params.npy',
          save_params_every=5,
          validate_every=5,
          # summary generation on the validation set
          generate_summary=False,
          summary_search_beam_size=2):
    params, tparams, x_embedder, y_embedder = init_params(
        model=model,
        corpus=corpus,
        optimizer=optimizer,
        learning_rate=learning_rate,
        embed_full_text_by=embed_full_text_by,
        seq_maxlen=seq_maxlen,
        summary_maxlen=summary_maxlen,
        summary_context_length=summary_context_length,
        internal_representation_dim=internal_representation_dim,
        attention_weight_max_roll=attention_weight_max_roll,
        l2_penalty_coeff=l2_penalty_coeff,
        train_split=train_split,
        epochs=epochs,
        minibatch_size=minibatch_size,
        seed=seed,
        dropout_rate=dropout_rate,
        summary_search_beam_size=summary_search_beam_size
        )

    # minibatch of encoded texts
    # size batchsize-by-seq_maxlen
    x = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')
    x_mask = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')

    # summaries for the minibatch of texts
    y = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')
    y_mask = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')

    nll = training_model_tensors(x, y, x_mask, y_mask, 
            params, tparams, y_embedder)
    cost = nll.mean()

    tparams_to_optimise = {key: tparams[key] for key in tparams
                           if not key.endswith('emb')}
    cost += params['l2_penalty_coeff'] * sum([(p ** 2).sum()
                                              for k, p in tparams_to_optimise.items()])
    inputs = [x, y, x_lengths, y_lengths]
    

    #trng, use_noise, \
    #    x, x_mask, y, y_mask, \
    #    opt_ret, \
    #    cost = \
    #    build_model(params, tparams, options)


        # after all regularizers - compile the computational graph for cost
    print('Building f_cost...')
    f_cost = theano.function(inputs, cost)
    print('Done')

    print('Computing gradient...')
    grads = T.grad(cost, list(tparams_to_optimise.values()))
    print('Done')

    # compile the optimizer, the actual computational graph is compiled here
    lr = T.scalar(name='lr')
    print('Building optimizers...')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams_to_optimise, grads, inputs, cost)
    print('Done')

    print('Building summary candidate token generator...')
    f_best_candidates = tfunc_best_candidate_tokens(params, tparams)
    print('Done')

    print('Loading corpus...')
    x_train, x_test, y_train, y_test, \
        x_lengths_train, x_lengths_test, \
        y_lengths_train, y_lengths_test \
        = load_corpus(params, tparams, embedding_x, embedding_y)
    n_train_batches = int(x_train.shape[0] / params['minibatch_size'])
    n_test_batches = int(x_test.shape[0] / params['minibatch_size'])
    print('Done')

    print('Optimization')
    for epoch in range(epochs):
        print('Epoch', epoch)

        params['phase'] = 'training'
        for batch_id in range(n_train_batches):
            print('Batch', batch_id)
            #use_noise.set_value(1.)
            
            # compute cost, grads and copy grads to shared variables
            current_batch = range(batch_id * params['minibatch_size'],
                                  (batch_id + 1) * params['minibatch_size'])
            cost = f_grad_shared(x_train[current_batch, :], y_train[current_batch, :], 
                                 x_lengths_train[current_batch], y_lengths_train[current_batch])
            print('cost', cost)

            # do the update on parameters
            f_update(learning_rate)

            #summary_token_ids = generate_summary(x_test[0, :], x_lengths_test[0], 
            #                                     f_best_candidates, params, tparams, 
            #                                     embedding_y)
            #print(embedding_y.documentFromVector(summary_token_ids))
        
        # save the params
        if epoch % params['save_params_every'] == 0:
            print('Saving...')
            save_params_(params, tparams, save_params)
            print('Done')

        # validating
        # compute the metrics and generate summaries (if requested)
        params['phase'] = 'test'
        if epoch % params['validate_every'] == 0:
            validate(params, tparams)

            for i in range(5):
                summary_token_ids = summarize(x_test[i, :], x_lengths_test[i], 
                                              f_best_candidates, params, tparams, 
                                              embedding_y)
                print(embedding_y.documentFromVector(summary_token_ids))
                print(embedding_y.documentFromVector(y_test[i, :])[:20])


