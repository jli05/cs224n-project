from __future__ import (division, absolute_import,
                        print_function, unicode_literals)
import os
import sys
import argparse
import logging
from ass_theano import train

def main():
    parser = argparse.ArgumentParser(
        description='Abstractive Sentence Summariser'
    )
    parser.add_argument('--log',
                        default='DEBUG',
                        choices=['DEBUG', 'WARNING', 'ERROR'],
                        help='log level for Python logging')
    parser.add_argument('--model',
                        default='baseline',
                        choices=['baseline', 'attention'],
                        help='model name')
    parser.add_argument('--corpus', required=True,
                        help='directory of the corpus for training and validation')

    parser.add_argument('--optimizer', 
                        default='adam',
                        choices=['adam', 'adadelta', 'rmsprop', 'sgd'],
                        help='optimizing algorithm')
    parser.add_argument('--learning-rate', type=float,
                        default=0.001,
                        help='learning rate for the optimizer')

    parser.add_argument('--summary-context-length', type=int,
                        default=5,
                        help='summary context length used for training')
    parser.add_argument('--L2-penalty-coeff', type=float,
                        default=0.0,
                        help='penalty coefficient to the L2-norms of the model params')
    parser.add_argument('--minibatch-size', type=int,
                        default=20,
                        help='mini batch size')
    parser.add_argument('--epochs', type=int,
                        default=10000,
                        help='number of epochs for training')
    parser.add_argument('--train-split', type=float,
                        default=0.75,
                        help='weight of training corpus in the entire corpus, the rest for validation')
    parser.add_argument('--seed', type=int,
                        default=None,
                        help='seed for the random stream')

    parser.add_argument('--dropout-rate', type=float,
                        default=None,
                        help='dropout rate in (0,1)')

    parser.add_argument('--embed-full-text-by',
                        choices=['word', 'sentence'],
                        default='word',
                        help='embed full text by word or sentence')
    parser.add_argument('--internal-representation-dim', type=int,
                        default=1000,
                        help='internal representation dimension')

    parser.add_argument('--attention-weight-max-roll', type=int,
                        default=1,
                        help='max roll for the attention weight vector in attention encoder')

    parser.add_argument('--load-params',
                        default=None,
                        help='file to load params from')

    parser.add_argument('--save-params',
                        default='ass_params.npy',
                        help='file for saving params')
    parser.add_argument('--save-params-every', type=int,
                        default=5,
                        help='save params every <k> epochs')
    parser.add_argument('--validate-every', type=int,
                        default=5,
                        help='validate every <k> epochs')
    
    parser.add_argument('--generate-summary',
                        action='store_true',
                        default=False,
                        help='whether to generate summaries when validating')
    parser.add_argument('--summary-maxlen', type=int,
                        default=500,
                        help='max length of each summary')
    parser.add_argument('--summary-search-beam-size', type=int,
                        default=2,
                        help='beam size for the summary search')
    
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())
    assert args.learning_rate > 0
    assert args.summary_context_length > 0
    assert args.L2_penalty_coeff >= 0
    assert args.minibatch_size > 0
    assert args.epochs >= 0
    assert (args.train_split > 0 and args.train_split <= 1)
    assert (args.seed is None or args.seed >= 0)
    assert (args.dropout_rate is None
            or (args.dropout_rate > 0 and args.dropout_rate < 1))
    assert args.internal_representation_dim > 0
    assert args.attention_weight_max_roll >= 0
    assert args.save_params_every > 0
    assert args.validate_every > 0
    assert args.summary_maxlen > 0
    assert args.summary_search_beam_size > 0
    
    train(
        model=args.model,
        corpus=args.corpus,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        summary_context_length=args.summary_context_length,
        l2_penalty_coeff=args.L2_penalty_coeff,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        train_split=args.train_split,
        seed=args.seed,
        dropout_rate=args.dropout_rate,
        embed_full_text_by=args.embed_full_text_by,
        internal_representation_dim=args.internal_representation_dim,
        attention_weight_max_roll=args.attention_weight_max_roll,
        load_params=args.load_params,
        save_params=args.save_params,
        save_params_every=args.save_params_every,
        validate_every=args.validate_every,
        generate_summary=args.generate_summary,
        summary_maxlen=args.summary_maxlen,
        summary_search_beam_size=args.summary_search_beam_size
    )

if __name__ == '__main__':
    main()
