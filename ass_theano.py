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


def get_encoder(model):
    def baseline_encoder(x, y, text_length, summary_length,
                         params, tparams):
        ''' baseline context encoder given one piece of text

        '''
        x_embedding = tparams['Xemb'][x[:text_length].flatten(), :]
        return x_embedding.mean(axis=0)

    def attention_encoder(x, y, text_length, summary_length,
                          params, tparams):
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

def conditional_distribution(x, y, text_length, summary_length,
                             params, tparams):
    ''' Return the conditional distribution of next summary word index

    Given the input text tensor and summary tensor, returns the distribution for the next summary word index
    '''
    encoder = get_encoder(params['model'])
    C = params['summary_context_length']
    y_embedding = tparams['Yemb'][y[(summary_length - C):summary_length].flatten(), :]
    h = T.tanh(T.dot(tparams['U'], y_embedding.flatten()))
    enc = encoder(x, y, text_length, summary_length, params, tparams)
    return T.nnet.softmax(T.dot(tparams['V'], h)
                          + T.dot(tparams['W'], enc)).flatten()

def conditional_score(x, y, text_length, j,
                      params, tparams):
    ''' Return conditional score of the (j+1)-th word index of the summary i.e. y[j]

    '''
    dist = conditional_distribution(x, y, text_length, j,
                                    params, tparams)
    return dist[y[j]]

def tfunc_best_candidate_tokens(params, tparams):
    x = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    x_length = T.cast(T.scalar(dtype=theano.config.floatX), 'int32')
    y = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    y_length = T.cast(T.scalar(dtype=theano.config.floatX), 'int32')
    
    dist = conditional_distribution(x, y, x_length, y_length, params, tparams)
    k = params['summary_search_beam_size']
    best_candidate_ids = dist.argsort()[-k:] 
    f = theano.function([x, y, x_length, y_length],
                        [best_candidate_ids, dist[best_candidate_ids]])
    return f

def summarize(x, text_length, f_best_candidates, params, tparams, embedding_y):
    ''' Generate summary for a single piece of text

    Uses beam search for size k
    '''
    id_pad = embedding_y.word_to_id[embedding_y.pad]
    C = params['summary_context_length']
    y = [id_pad] * C
    beams = [(0.0, y)]

    k = params['summary_search_beam_size']
    for j in range(params['summary_maxlen']):
        new_beams = []
        for (base_score, y) in beams:
            candidate_ids, candidate_prob = f_best_candidates(x, y, text_length, len(y)) 
            for (token_id, token_prob) in zip(candidate_ids, candidate_prob):
                heapq.heappush(new_beams, (base_score - np.log(token_prob), y + [token_id]))
        beams = heapq.nsmallest(k, new_beams)

    (best_nll_score, summary) = heapq.heappop(beams)
    return summary[C:]


def training_model_tensors(params, tparams):
    ''' Return tensors for training model

    '''
    # word/sentence indices for full original text
    # could be word-by-word, sentence-by-sentence, etc
    # size batchsize-by-max_l
    # max_l is the max number of language units (word, sentence, paragraph) of a text
    x = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')
    x_lengths = T.cast(T.vector(dtype=theano.config.floatX), 'int32')

    # word indices for the summaries
    # size batchsize-by-max_m
    # max_m is the max number of words in the summary
    y = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')
    y_lengths = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    #y_lengths = T.ivector()

    # predicted probs
    # size batchsize-by-(num-contexts in the summary)
    prob = T.zeros_like(y)
    prob_mask = T.zeros_like(y)
    C = params['summary_context_length']

    fn = lambda j, x, y, x_len: conditional_score(x, y, x_len, j, params, tparams)
    for i in range(params['minibatch_size']):
        range_ = T.arange(C, y_lengths[i], dtype='int32')

        prob_mask = T.set_subtensor(prob_mask[i, range_], 1)
        
        prob_, _ = theano.scan(fn,
                               sequences=[range_],
                               outputs_info=None,
                               non_sequences=[x[i, :], y[i, :], x_lengths[i]],
                               n_steps=y_lengths[i] - C)
                            
        prob = T.set_subtensor(prob[i, range_], prob_.flatten())

    nll = - T.log(prob + 1e-16) * prob_mask    
    return x, y, x_lengths, y_lengths, nll

def load_params_(params, tparams, file_path):
    pass

def save_params_(params, tparams, file_path):
    pass

def init_shared_tparam_(name, shape, value=None,
                        borrow=True, dtype=theano.config.floatX):
    if value is None:
        value=np.random.uniform(low=-0.1, high=0.1, size=shape)
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
          optimizer='adam',
          learning_rate=0.001,
          # training params
          summary_context_length=10,
          l2_penalty_coeff=0.0,
          minibatch_size=20,
          epochs=float('inf'),
          train_split=0.75,
          seed=None,
          dropout_rate=None,
          embed_full_text_by='word',
          internal_representation_dim=1000,
          attention_weight_max_roll=5,
          # model load/save
          load_params=None,
          save_params='ass_params.npy',
          save_params_every=5,
          validate_every=5,
          # summary generation on the validation set
          generate_summary=False,
          summary_maxlen=500,
          summary_search_beam_size=2):
    if load_params is not None:
        params, tparams, embedding_x, embedding_y = load_params_(load_params)
        params.update({'corpus': corpus,
                       'epochs': epochs,
                       'train_split': train_split,
                       'generate_summary': generate_summary,
                       'summary_maxlen': summary_maxlen,
                       'summary_search_beam_size': summary_search_beam_size})
    else:
        params, tparams, embedding_x, embedding_y = init_params(
            model,
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
            validate_every
            )

    x, y, x_lengths, y_lengths, nll = training_model_tensors(params, tparams) 
    cost = nll.sum(axis=1).mean()
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


