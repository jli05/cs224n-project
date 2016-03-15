from __future__ import (division, absolute_import,
                        print_function, unicode_literals)
import argparse
import logging
import sys
import os
import glob
import json
import pickle
import heapq
import numpy as np
from random import sample
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from optimisers import adam, adadelta, rmsprop, sgd
from embeddings import gloveDocumentParser
from sklearn.cross_validation import train_test_split

EPSILON_FOR_LOG = 1e-8

def get_encoder(context_encoder):
    def baseline_encoder(x, y, x_mask, y_pos, params, tparams):
        ''' baseline context encoder given one piece of text

        Returns ctx each row for a training instance
        '''
        if x.ndim == 1:
            mb_size = 1
        elif x.ndim == 2:
            mb_size = x.shape[0]
        seq_len = params['seq_maxlen']
        wv_size = params['full_text_word_vector_size']
        
        x_emb = tparams['Xemb'][x.flatten(), :]
        x_emb_masked = T.batched_dot(x_emb, x_mask.flatten())

        if x.ndim == 1:
            ctx = x_emb_masked.sum(axis=0) / x_mask.sum()
        elif x.ndim == 2:
            ctx = T.batched_dot(
                x_emb_masked.reshape((mb_size, seq_len, wv_size)).sum(axis=1),
                1 / x_mask.sum(axis=1)
            )
         
        return T.cast(ctx, theano.config.floatX) 

    def attention_encoder(x, y, x_mask, y_pos, params, tparams):
        ''' attention-based context encoder given one piece of text

        '''
        C = params['summary_context_length']
        Q = params['attention_weight_max_roll']
        wv_size_x = params['full_text_word_vector_size']
        wv_size_y = params['summary_word_vector_size']
        P = tparams['att_P']
        m = tparams['att_P_conv']

        if x.ndim == 1:
            x_emb = tparams['Xemb'][x, :]
            y_emb = tparams['Yemb'][y[(y_pos - C):y_pos], :]
            p = T.nnet.softmax(
                T.dot(x_emb, T.dot(P, y_emb.flatten()))
            )
            p_masked = p * x_mask
            p_masked /= p_masked.sum()
            ctx = T.dot(x_emb, T.dot(m, p_masked))

        elif x.ndim == 2:
            x_emb = tparams['Xemb'][x.flatten(), :]
            x_emb = x_emb.reshape((x.shape[0], x.shape[1], wv_size_x))
            y_emb = tparams['Yemb'][y[:, (y_pos - C):y_pos].flatten(), :]
            y_emb = y_emb.flatten().reshape((x.shape[0], C * wv_size_y)).T
            p = T.nnet.softmax(
                    T.batched_dot(x_emb, T.dot(P, y_emb).T)
                    )
            p_masked = p * x_mask
            p_masked /= p_masked.sum(axis=1)
            ctx = T.batched_dot(x_emb, T.dot(m, p_masked.T).T) 

        return T.cast(ctx, theano.config.floatX)


    if context_encoder == 'baseline':
        return baseline_encoder
    elif context_encoder == 'attention':
        return  attention_encoder
    else:
        raise ValueError('Invalide context encoder {:}'.format(context_encoder))

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
    enc = get_encoder(params['context_encoder'])
    C = params['summary_context_length']
    wv_size = params['summary_word_vector_size']

    if x.ndim == 1:
        y_emb = tparams['Yemb'][y[(y_pos - C):y_pos].flatten(), :].flatten()
        h = T.tanh(T.dot(tparams['U'], y_emb) + tparams['b']).flatten()
        ctx = enc(x, y, x_mask, y_pos, params, tparams)

        u = T.dot(tparams['V'], h) + T.dot(tparams['W'], ctx)
        y_next = T.nnet.softmax(u).flatten()

    elif x.ndim == 2:
        mb_size = x.shape[0]
        y_emb = tparams['Yemb'][y[:, (y_pos - C):y_pos].flatten(), :]
        y_emb = y_emb.flatten().reshape((C * wv_size, mb_size))
        # each row for a training instance
        # (in order to broadcast the vector b along the row axis)
        h = T.tanh((T.dot(tparams['U'], y_emb)).T + tparams['b'])
        # each row for a training instance
        ctx = enc(x, y, x_mask, y_pos, params, tparams)

        # each column for a training instance
        u = T.dot(tparams['V'], h.T) + T.dot(tparams['W'], ctx.T)
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

def training_model_output(x, y, x_mask, y_mask, params, tparams, y_embedder):
    ''' Return tensors for training model

    '''
    mb_size = params['minibatch_size']
    C = params['summary_context_length']
    l = params['summary_maxlen']
    
    # pad y
    id_pad = y_embedder.word_to_id[y_embedder.pad]
    y_padded = T.concatenate([T.alloc(id_pad, mb_size, C), y], axis=1)

    # compute the model probabilities for each encoded token in y
    fn = lambda y_pos, x, y, x_mask: conditional_score(x, y, x_mask, y_pos, params, tparams)
    y_pos_range = T.arange(C, l + C, dtype='int32')

    prob_, _ = theano.scan(fn,
                           sequences=y_pos_range,
                           non_sequences=[x, y_padded, x_mask],
                           n_steps=l)
    #prob = T.concatenate([v.reshape((mb_size, 2)) for v in prob_], axis=1)
    prob = prob_.T

    # masked negative log-likelihood
    nll_per_token = - T.log(prob + EPSILON_FOR_LOG) * y_mask
    nll_per_text = T.sum(nll_per_token, axis=1) / T.sum(y_mask, axis=1) 
    return T.cast(nll_per_text, theano.config.floatX)

def tfunc_best_candidate_tokens(params, tparams):
    ''' Returns a Theano function that computes the best k candidate terms for the next position in the summary

    '''
    k = params['summary_search_beam_size']

    x = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    x_mask = T.vector(dtype=theano.config.floatX)
    y = T.cast(T.vector(dtype=theano.config.floatX), 'int32')
    y_pos = T.cast(T.scalar(dtype=theano.config.floatX), 'int32')
    
    dist = conditional_distribution(x, y, x_mask, y_pos, params, tparams)
    best_candidate_ids = dist.argsort()[-k:] 
    f = theano.function([x, y, x_mask, y_pos],
                        [best_candidate_ids, dist[best_candidate_ids]],
                        allow_input_downcast=True)
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
    with open(file_path, 'rb') as f:
        params = pickle.load(f)
        tparams = pickle.load(f)

def save_params_(params, tparams, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(params, f)
        pickle.dump(tparams, f)

def init_params(**kwargs):
    def init_shared_tparam_(name, shape, value=None,
                            borrow=True, dtype=theano.config.floatX):
        if value is None:
            value=np.random.uniform(low=-0.02, high=0.02, size=shape)
        return theano.shared(value=value.astype(dtype), 
                             name=name,
                             borrow=borrow)

    def attention_prob_conv_matrix(Q, l):
        assert l >= Q
        m = np.diagflat([1.0] * l)
        for i in range(1, Q):
            m += np.diagflat([1.0] * (l - i), k=i)
            m += np.diagflat([1.0] * (l - i), k=-i)
        m = m / np.sum(m, axis=0)
        return m

    params = kwargs.copy()
    params.update({'rng': np.random.RandomState(seed=params['seed']),
                   'trng': RandomStreams(seed=params['seed'])}) 
    
    if params['embed_full_text_by'] == 'word':
        x_embedder = gloveDocumentParser('glove/glove.10k.300d.txt')
        y_embedder = x_embedder 
    else:
        x_embedder = None
        y_embedder = None
    params.update({'full_text_word_vector_size': x_embedder.token_dim,
                   'summary_word_vector_size': y_embedder.token_dim})

    h = params['internal_representation_dim']
    C = params['summary_context_length']
    l = params['seq_maxlen']
    V_x = x_embedder.embedding_n_tokens
    V_y = y_embedder.embedding_n_tokens
    d_x = x_embedder.token_dim    # full text word vector size
    d_y = y_embedder.token_dim    # summary word vector size

    tparams = {
        'U': init_shared_tparam_('U', (h, C * d_y)),
        'b': init_shared_tparam_('b', (h,)),
        'V': init_shared_tparam_('V', (V_y, h)), 
        'W': init_shared_tparam_('W', (V_y, d_x)), 
        'Xemb': init_shared_tparam_('Xemb', (V_x, d_x), 
                                    value=x_embedder.word_to_vector_matrix),
        'Yemb': init_shared_tparam_('Yemb', (V_y, d_y),
                                    value=y_embedder.word_to_vector_matrix)
        }
    if params['context_encoder'] == 'attention':
        Q = params['attention_weight_max_roll']
        m = attention_prob_conv_matrix(Q, l)
        tparams.update({
            'att_P': init_shared_tparam_('att_P', (d_x, C * d_y)),
            'att_P_conv': init_shared_tparam_('att_P_conv', (l, l),
                                              value=m)
            })
    
    return params, tparams, x_embedder, y_embedder

def load_corpus(params, tparams, x_embedder, y_embedder):
    def pad_to_length(v, pad, l):
        return np.pad(v, (0, l - len(v)), 'constant', 
                      constant_values=(pad, pad))

    def mask_vector(v, l):
        return [1] * len(v) + [0] * (l - len(v))

    C = params['summary_context_length']
    l_x = params['seq_maxlen']
    l_y = params['summary_maxlen']
    id_pad_x = x_embedder.word_to_id[x_embedder.pad]
    id_pad_y = y_embedder.word_to_id[y_embedder.pad]

    x_ = []
    y_ = []
    x_mask_ = []
    y_mask_ = []
    for file_path in glob.iglob(os.path.join(params['corpus'], '*.json')):
        try:
            with open(file_path, 'r') as f:
                document = json.load(f)
            full_text_vector = x_embedder.parseDocument(document['full_text'])
            summary_vector = y_embedder.parseDocument(document['summary'])
        
            if not len(full_text_vector) or not len(summary_vector):
                continue
            
            x_.append(pad_to_length(full_text_vector[:l_x], id_pad_x, l_x))
            y_.append(pad_to_length(summary_vector[:l_y], id_pad_y, l_y))
            x_mask_.append(mask_vector(full_text_vector[:l_x], l_x))
            y_mask_.append(mask_vector(summary_vector[:l_y], l_y))
        except Exception as e:
            continue
    print('Loaded {:} files'.format(len(x_)))
    
    x = np.array(x_, dtype='int32')
    y = np.array(y_, dtype='int32')
    x_mask = np.array(x_mask_)
    y_mask = np.array(y_mask_)

    x_train, x_test, y_train, y_test, \
        x_mask_train, x_mask_test, \
        y_mask_train, y_mask_test = \
        train_test_split(x, y, x_mask, y_mask,
                         train_size=params['train_split'],
                         random_state=params['rng'])
    
    return x_train, x_test, y_train, y_test, \
        x_mask_train, x_mask_test, \
        y_mask_train, y_mask_test

def train(context_encoder='baseline',
          corpus=None,
          # optimiser
          optimizer='adam',
          learning_rate=0.001,
          # model params
          embed_full_text_by='word',
          seq_maxlen=500,
          summary_maxlen=200,
          summary_context_length=10,
          internal_representation_dim=2000,
          attention_weight_max_roll=5,
          # training params
          l2_penalty_coeff=0.0,
          train_split=0.75,
          epochs=float('inf'),
          minibatch_size=20,
          seed=None,
          dropout_rate=None,
          # model load/save
          save_params='ass_params.pkl',
          save_params_every=5,
          validate_every=5,
          print_every=5,
          # summary generation on the validation set
          generate_summary=False,
          summary_search_beam_size=2):
    params, tparams, x_embedder, y_embedder = init_params(
        context_encoder=context_encoder,
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
    x_mask = T.matrix(dtype=theano.config.floatX)

    # summaries for the minibatch of texts
    y = T.cast(T.matrix(dtype=theano.config.floatX), 'int32')
    y_mask = T.matrix(dtype=theano.config.floatX)

    nll = training_model_output(x, y, x_mask, y_mask, 
            params, tparams, y_embedder)
    cost = nll.mean()

    tparams_to_optimise = {key: tparams[key] for key in tparams
                           if not key.endswith('emb')}
    cost += params['l2_penalty_coeff'] * sum([(p ** 2).sum()
                                              for k, p in tparams_to_optimise.items()])
    inputs = [x, y, x_mask, y_mask]
    
    # after all regularizers - compile the computational graph for cost
    print('Building f_cost... ', end='')
    f_cost = theano.function(inputs, cost, allow_input_downcast=True)
    print('Done')

    print('Computing gradient... ', end='')
    grads = T.grad(cost, list(tparams_to_optimise.values()))
    print('Done')

    # compile the optimizer, the actual computational graph is compiled here
    lr = T.scalar(name='lr')
    print('Building optimizers... ', end='')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams_to_optimise, grads, inputs, cost)
    print('Done')

    print('Building summary candidate token generator... ', end='')
    f_best_candidates = tfunc_best_candidate_tokens(params, tparams)
    print('Done')

    print('Loading corpus... ', end='')
    x_train, x_test, y_train, y_test, \
        x_mask_train, x_mask_test, \
        y_mask_train, y_mask_test \
        = load_corpus(params, tparams, x_embedder, y_embedder)
    n_train_batches = int(x_train.shape[0] / params['minibatch_size'])
    n_test_batches = int(x_test.shape[0] / params['minibatch_size'])
    print('Done')

    print('Optimization')
    test_ids_to_summarize = sample(range(x_test.shape[0]), 5) 
    for epoch in range(epochs):
        print('Epoch', epoch)

        # training of all minibatches
        params['phase'] = 'training'
        training_costs = []
        for batch_id in range(n_train_batches):
            if batch_id % print_every == 0:
                print('Batch {:} '.format(batch_id), end='')
            # compute cost, grads and copy grads to shared variables
            #use_noise.set_value(1.)
            current_batch = range(batch_id * params['minibatch_size'],
                                  (batch_id + 1) * params['minibatch_size'])
            cost = f_grad_shared(x_train[current_batch, :], 
                                 y_train[current_batch, :], 
                                 x_mask_train[current_batch, :], 
                                 y_mask_train[current_batch, :])
            cost = np.asscalar(cost)
            training_costs.append(cost)
            # do the update on parameters
            f_update(learning_rate)
            if batch_id % print_every == 0:
                print('Cost {:.4f}'.format(cost))
        print('Epoch {:} mean training cost {:.4f}'.format(
            epoch, np.mean(training_costs)
            ))

        # save the params
        if epoch % save_params_every == 0:
            print('Saving... ', end='')
            save_params_(params, tparams, save_params)
            print('Done')

        # validate
        # compute the metrics and generate summaries (if requested)
        params['phase'] = 'test'
        if epoch % validate_every == 0:
            print('Validating')
            validate_costs = []
            for batch_id in range(n_test_batches):
                if batch_id % print_every == 0:
                    print('Batch {:} '.format(batch_id), end='')
                current_batch = range(batch_id * params['minibatch_size'],
                                      (batch_id + 1) * params['minibatch_size'])
                validate_cost = f_cost(x_test[current_batch, :], 
                                       y_test[current_batch, :], 
                                       x_mask_test[current_batch, :], 
                                       y_mask_test[current_batch, :])
                validate_cost = np.asscalar(validate_cost)
                validate_costs.append(validate_cost)
                if batch_id % print_every == 0:
                    print('Validation cost {:.4f}'.format(validate_cost))
            print('Epoch {:} mean validation cost {:.4f}'.format(
                  epoch, np.mean(validate_costs)
                  ))

            if generate_summary:
                print('Generating summary')
                for i in test_ids_to_summarize:
                    summary_token_ids = summarize(
                        x_test[i, :].flatten(), x_mask_test[i, :].flatten(), 
                        f_best_candidates, 
                        params, tparams,
                        y_embedder)
                    print('Sample :', y_embedder.documentFromVector(summary_token_ids))
                    print('Truth :', y_embedder.documentFromVector(y_test[i, :])[:20])


