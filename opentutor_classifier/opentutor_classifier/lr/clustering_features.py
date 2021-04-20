from typing import Dict, List
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from .features import word_alignment_feature, number_of_negatives

import numpy as np
import pandas as pd

from itertools import combinations
from gensim.models.keyedvectors import Word2VecKeyedVectors


class CustomAgglomerativeClustering():
  def __init__(self, word2vec : Word2VecKeyedVectors, index2word_set):
    self.data = None
    self.word2vec = word2vec
    self.index2word_set = index2word_set

  def alignment_metric(self, x, y) -> float:
    if len(self.data[int(x[0])]) > len(self.data[int(y[0])]): x,y = y,x
    return ( 1-word_alignment_feature( self.data[int(y[0])], self.data[int(x[0])], self.word2vec, self.index2word_set) )

  def fit_predict(self, data):
    self.data = data
    X = np.arange(len(self.data)).reshape(-1, 1)

    # Calculate pairwise distances with the new metric.
    m = pairwise_distances(X, X, metric=self.alignment_metric)

    agg = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
    return agg.fit_predict(m)


def get_clusters( good_answers : np.array, bad_answers: np.array, word2vec : Word2VecKeyedVectors, index2word_set):
    good_labels = CustomAgglomerativeClustering(word2vec, index2word_set).fit_predict( good_answers)
    bad_labels = CustomAgglomerativeClustering(word2vec, index2word_set).fit_predict( bad_answers )
    return good_labels, bad_labels

def get_best_candidate( sentence_cluster: np.array, word2vec : Word2VecKeyedVectors, 
                index2word_set, cuttoff_length : int = 15) -> str:
    
    sentence_cluster = sentence_cluster[np.vectorize( lambda x: len(x) < cuttoff_length )( sentence_cluster )]
    if len(sentence_cluster) < 5: return ""

    avg_proximity = np.zeros( len(sentence_cluster) )
    for i, row1 in enumerate( sentence_cluster ):
      for row2 in sentence_cluster:
        if row1 == row2: continue
        avg_proximity[i] += ((len(row2)/(len(row2)+len(row1))) * word_alignment_feature(row2, row1, word2vec, index2word_set)) #check best alignment in both directions
        avg_proximity[i] += ((len(row1)/(len(row2)+len(row1))) * word_alignment_feature(row1, row2, word2vec, index2word_set))
 
    avg_proximity /= len(sentence_cluster)
    best_idx = np.argmax( avg_proximity )
    return sentence_cluster[best_idx]

def generate_patterns_from_candidates( data: pd.DataFrame, best_targets: Dict[str, str] ):
    useful_pattern_for_each_cluster = { 'good':[], 'bad':[] }
    for label, best_target in best_targets:
        words = set()
        for word in best_target:
          if number_of_negatives([word])[0] > 0: words.add('[NEG]')
          else: words.add(word)
        
        for word in words:
          data[word] = data['[SENTENCES]'].apply( lambda x: 1*(word in x) )
        
        data['[NEG]'] = data['[SENTENCES]'].apply( lambda x: 1*(number_of_negatives(x)[0]>0) )
        
        #maximum pattern length will be 4
        total_patterns = list(words)
        for i in range( 2, min( len(words)+1, 4) ):
            combs = combinations( words, i )
            for comb in combs:
                comb = sorted(list(comb))
                data[" + ".join(comb)] = 1
                for word in comb: data[" + ".join(comb)]*=data[word]
                total_patterns.append( " + ".join(comb) )
        useful_pattern_for_each_cluster[label].extend( total_patterns )
    return data, useful_pattern_for_each_cluster

def generate_feature_candidates(good_answers: np.array, bad_answers: np.array, 
          word2vec : Word2VecKeyedVectors, index2word_set): 

    good_answers, bad_answers = np.array(good_answers), np.array(bad_answers)
    good_labels, bad_labels = get_clusters(good_answers, bad_answers, word2vec, index2word_set )

    best_candidates = []
    best_candidates.append( ('good', get_best_candidate( good_answers[ good_labels == 0 ], word2vec, index2word_set) ) )
    best_candidates.append( ('good', get_best_candidate( good_answers[ good_labels == 1 ], word2vec, index2word_set) ) )
    best_candidates.append( ('bad', get_best_candidate( bad_answers[ bad_labels == 0 ], word2vec, index2word_set) ) )
    best_candidates.append( ('bad', get_best_candidate( bad_answers[ bad_labels == 1 ], word2vec, index2word_set) ) )

    data = pd.DataFrame( {"[SENTENCES]": list(good_answers)+list(bad_answers), '[LABELS]':[1]*len(good_answers) + [0]*len(bad_answers) } )
    
    data, candidates = generate_patterns_from_candidates( data,  best_candidates)
    return data, candidates

def select_feature_candidates( data: pd.DataFrame, candidates: List[str], fpr_cuttoff : float = 0.98 ) -> List[str]:
    useful_features = []
    for label in ("good", "bad"):
        good, bad, patterns = [], [], []
        for candidate in candidates[label]:
            good.append( np.sum( data[candidate]*data['[LABELS]']  ) )
            bad.append( np.sum( data[candidate]*(1-data['[LABELS]']) ) )
            patterns.append( candidate )
        good, bad, patterns = np.array(good), np.array(bad), np.array(patterns)
        one_fpr = None
        if label=='good': one_fpr = 1 - (bad / np.sum(1 - data['[LABELS]']))  #np.array(good)/(np.array(good)+np.array(bad)+1e-10)
        else: one_fpr = 1 - (good / np.sum(data['[LABELS]']) )

        patterns = patterns[ one_fpr > fpr_cuttoff ]
        one_fpr = one_fpr[ one_fpr > fpr_cuttoff ]
        useful_features.extend( list( patterns ) )
    return useful_features