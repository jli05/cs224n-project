from __future__ import (division, absolute_import,
                        print_function, unicode_literals)
import nltk
from nltk.util import ngrams
import os.path
from nltk.collocations import *
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
import sys
#sys.path.insert(0, "./skip-thoughts")
#import skipthoughts
import numpy as np

def _get_least_common_word_vector(num_vecs = 0, glove_filename="glove/glove.6B.50d.txt"):
    if num_vecs == 0:
        with open(glove_filename, 'r') as f:
                for line in f: pass
                vec = line.split(' ')[1:]
                return list(map(float, vec))
    else:
        total = 0
        with open(glove_filename, 'r') as f:
            for line in f: total += 1
        count = 0
        uncommon_vectors = []
        with open(glove_filename, 'r') as f:
            for line in f:
                if total - count < num_vecs:
                    uncommon_vectors.append(list(map(float, line.split(' ')[1:])))
                count += 1
        return np.mean(np.array(uncommon_vectors) , axis = 0)


def map_sentence_to_glove(sentence, glove_filename="glove/glove.6B.50d.txt", cache={}, default_vec = []):
        vectorized_sentence = []
        indexed_sentence = []
        for word in sentence:
                lcw = word.lower()
                index = None
                if lcw not in cache.keys():
                        with open(glove_filename, 'r') as f:
                                v = None
                                count = 1 # so we can make 0 UNK
                                for line in f:
                                        first_word = line.split(' ', 1)[0]
                                        if first_word == lcw:
                                                v = line.split(' ')[1:]
                                                v = map(float, v)
                                                cache[lcw] = v, count
                                                index = count
                                                break
                                        count += 1
                                if v is None:
                                    v = default_vec
                                    index = count
                else:
                        v, index = cache[lcw]
                vectorized_sentence.append(v)
                indexed_sentence.append(index)
        return vectorized_sentence, cache, indexed_sentence
        
class gloveDocumentParser(object):
        def __init__(self, glove_file_name, unk_size=100, pad_token = "--PAD--", unk_token = "--UNK--"):
                self.word_to_vector, self.word_to_id, self.id_to_word, \
                    self.vocab_size, vec_length = self.loadGloveFromFile(glove_file_name)
                padding_vector = [0] * vec_length
                PAD = pad_token 
                UNK = unk_token 

                self.unk = UNK
                self.pad = PAD
                self.word_to_vector[UNK] = _get_least_common_word_vector(unk_size, glove_filename=glove_file_name)
                self.word_to_id[UNK] = 0
                self.id_to_word[0] = UNK 

                self.word_to_vector[PAD] = padding_vector
                self.word_to_id[PAD] = 1
                self.id_to_word[1] = PAD 
                self.word_to_vector_matrix = []
                for i in range(self.vocab_size):
                        self.word_to_vector_matrix.append( self.word_to_vector[ self.id_to_word[i] ])
                self.word_to_vector_matrix = np.array( self.word_to_vector_matrix )


                self.embedding_n_tokens = self.word_to_vector_matrix.shape[0]
                self.token_dim = self.word_to_vector_matrix.shape[1]

        def loadGloveFromFile(self, glove_file_name):
                word_to_vec = {}
                word_to_id = {}
                id_to_word = {}
                index = 2 
                with open(glove_file_name, 'r') as f:
                        for line in f:
                                split_line = line.split(' ')
                                word_key = split_line[0]
                                vector = list(map(float, split_line[1:]))
                                vector_dim = len(vector)
                                word_to_vec[word_key] = vector
                                word_to_id[word_key] = index
                                id_to_word[index] = word_key
                                index += 1
                return word_to_vec, word_to_id, id_to_word, index, vector_dim

        def parseDocument(self, document):
            tokens = wordpunct_tokenize(document)
            index_matrix = [self.word_to_id[token.lower()] if token.lower() in self.word_to_id
                    else self.word_to_id[self.unk] for token in tokens]
            return index_matrix

        def documentFromVector(self, id_vector):
            doc = [self.id_to_word[_id] for _id in id_vector]
            return doc

def rouge_score(reference, hypothesis, n):
    ref_ngrams = list(ngrams(wordpunct_tokenize(reference), n))
    hyp_ngrams = list(ngrams(wordpunct_tokenize(hypothesis), n))
    matching_ngrams = [x for x in hyp_ngrams if x in ref_ngrams]
    return 1.0 * len(matching_ngrams) / len(ref_ngrams) 

class SkipThoughts(object):
        def __init__(self, num_sentences=10):
                self.model = skipthoughts.load_model()
                self.word_to_id = {}
                self.id_to_word = {}
                self.next_mapped_index = 1
                self.next_row = 0
                self.emb_matrix = np.zeros( (num_sentences, 4800) )

        def parseDocument(self, text):
                sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = sent_tokenizer.tokenize(text)
                vectors = self.vectorize(sentences)
                index_vector = []
                for sentence, vector in zip(sentences, vectors):
                    _id = None
                    if sentence not in self.word_to_id.keys():
                        self.word_to_id[sentence] = self.next_mapped_index
                        self.id_to_word[self.next_mapped_index] = sentence
                        _id = self.next_mapped_index
                        self.next_mapped_index += 1
                    else: _id = self.word_to_id[sentence]
                    self.emb_matrix[self.next_row, :] = np.array( vector )
                    self.next_row += 1
                    index_vector.append( _id )
                return self.emb_matrix[:self.next_row, :], index_vector

        def reset(self):
            self.word_to_id = {}
            self.id_to_word = {}
            self.next_mapped_index = 1
            self.next_row = 0

        def vectorize(self, list_of_sentences):
                return skipthoughts.encode(self.model, list_of_sentences)

def rouge_test():
    reference = "The book was very good"
    hyp = "The book was very interesting"
    print(rouge_score(reference, hyp, 1))

def doc_parser_test():
        g = gloveDocumentParser("glove.6B.50d.txt")
        print(g.parseDocument("This is a sentence. This is another sentence. What can you do about it, glove?"))
        print(g.documentFromVector([39, 16, 9, 2424, 4]))

def skip_thoughts_test():
    s = SkipThoughts()
    a, b = s.parseDocument("This is a sentence. This is another sentence. Hopefully, we get a good matrix representation of this!")
    print(a)
    print(b)
    print(s.emb_matrix)

def UNKtest():
        w = word2vec()
        w.generateUNK()

if __name__=="__main__":
        #doc_parser_test()
        #rouge_test()
        skip_thoughts_test()
