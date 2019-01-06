from __future__ import division
import matplotlib.pyplot as plt

import numpy as np
import sys
import math
from collections import Counter
import time
import help_funcs as ut
import EM_algorithm as em

def ex3_process(articles_file, topics):
    headers_train_data, articles_train_data, all_words_with_all_freqs, articles_with_their_words_freqs = ut.create_train_data(
            articles_file)
    topics_list = ut.get_topics(topics)
    words_clusters = ut.split_into_clusters(articles_train_data)

    # Run the em algorithm to find the best wti for all docs according to the train data with the specific parameters
    # we gave
    final_weights = em.em_process(articles_with_their_words_freqs, all_words_with_all_freqs, words_clusters, len(topics_list))
    #Create the conf matrix from the best weights
    conf_matrix, clusters_with_topics, documents_in_clusters = ut.create_confusion_matrix(final_weights, articles_with_their_words_freqs,topics_list, headers_train_data)
    print conf_matrix
    docs_with_classification = ut.assign_classifications_to_docs(clusters_with_topics,documents_in_clusters)
    print "\n"
    accuracy = ut.calc_accuracy(headers_train_data, docs_with_classification)
    print "accuracy - ", accuracy

if __name__ == "__main__":
    ex3_process("develop.txt", "topics.txt")