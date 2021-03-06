'''
Coral Malachi	314882853
Avishai Zagoury 209573633
'''
from __future__ import division

import help_funcs as ut
import EM_algorithm as em

def main(articles_file, topics):
    #get the topics
    topics_model = ut.get_the_topics_lst(topics)
    #get the training set
    headers, articles, words_freqs, articles_freqs = ut.make_train_set(
            articles_file)

    #divide the words into cluster
    words_into_clusters = ut.divide_clusters(articles)

    #get the good wights for our model
    w_model = em.run_em_algorithm(articles_freqs, words_freqs, words_into_clusters, len(topics_model))

    #create the confuision matrix
    conf_matrix, clusters_and_topics, articles_of_clusters = ut.make_conf_matrix(w_model, articles_freqs,topics_model, headers)
    # conf_matrix_descending_order = sorted(conf_matrix, key=lambda line: line[-1], reverse=True)
    print conf_matrix

    #add the topic to the articles
    articles_by_topic = ut.add_tag_to_articles(clusters_and_topics,articles_of_clusters)
    #print empty line
    print "\n"
    #compute the accuracy of the model
    accuracy = ut.compute_accuracy(headers, articles_by_topic)
    print "the accuracy of our model is- ", accuracy

if __name__ == "__main__":
    main("develop.txt", "topics.txt")
