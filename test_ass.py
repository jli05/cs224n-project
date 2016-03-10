import numpy as np
import theano
import theano.tensor as T
from ass_theano import *

import pytest


@pytest.fixture
def global_params():
    params = {
            'minibatch_size': 2,
            'seq_maxlen': 2,
            'word_vector_size': 2
            }
    tparams = {
            'Xemb': theano.shared(
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                name='Xemb'
                )
            }
    return params, tparams

def test_baseline_encoder_1():
    model = get_encoder('baseline')
    params, tparams = global_params()
    
    x = T.imatrix()
    x_mask = T.imatrix()
    ctx = model(x, None, 
                x_mask, None,
                params, tparams)

    f = theano.function([x, x_mask], ctx)    
    assert np.all(f([[0, 1], [2, 3]], [[1, 0], [1, 1]]) 
                    == np.array([[1, 2], [6, 7]]))

def test_baseline_encoder_2():
    model = get_encoder('baseline')
    params, tparams = global_params()
    
    x = T.ivector()
    x_mask = T.ivector()
    ctx = model(x, None, 
                x_mask, None,
                params, tparams)

    f = theano.function([x, x_mask], ctx) 
    assert np.all(f([0, 1, 2], [0, 1, 1]) == np.array([4, 5]))
