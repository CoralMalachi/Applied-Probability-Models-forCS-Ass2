'''
Coral Malachi	314882853
Avishai Zagoury 209573633
'''
from __future__ import division


import numpy as np

from collections import Counter

NUM_CLUSTERS = 9



def add_tag_to_articles(clusters,articles):
    tagged_articles = []
    for cluster_id in articles:
        for x in articles[cluster_id]:
            tag = clusters[cluster_id]
            tagged_articles.append((x, tag))
    return tagged_articles

def compute_accuracy(headers, articles_with_tags):
    """

    :param headers:
    :param articles_with_tags:
    :return: the accuracy of the model
    """
    accuracy = 0
    for article in articles_with_tags:
        if article[1] in headers[article[0]]:
            accuracy += 1

    return accuracy / len(headers)

def compute_lidstone(c_words, train_len, vocab_len, lambda_val):
    """

    :param c_words:
    :param train_len:
    :param vocab_len:
    :param lambda_val:
    :return:
    """
    p_lid = (c_words + lambda_val) / (train_len + lambda_val * vocab_len)
    return p_lid




def make_conf_matrix(model_w, articles_and_freq,model_topics, topics_of_cure_article):
    """

    :param model_w:
    :param articles_and_freq:
    :param model_topics:
    :param topics_of_cure_article:
    :return:
    """
    articles_of_clusters = {}
    clusters_and_topics = {}

    len_topics = len(model_topics)
    len_clusters = len(model_w[0].keys())

    #init the conf matrix
    confusion_matrix_to_return = np.zeros((len_clusters, len_topics + 1))


    for t in articles_and_freq:
        index_of_max = 0
        max_w = model_w[t][0]

        for i in range(0, len_clusters):
            if model_w[t][i] > max_w:
                max_w = model_w[t][i]
                index_of_max = i
        if index_of_max not in articles_of_clusters:
            articles_of_clusters[index_of_max] = []

        articles_of_clusters[index_of_max].append(t)



    for x in range(0, len_clusters):
        for y in range(0, len_topics):
            cure_topic = model_topics[y]
            for t in articles_of_clusters[x]:
                if cure_topic in topics_of_cure_article[t]:
                    #update count value
                    confusion_matrix_to_return[x][y] += 1

        confusion_matrix_to_return[x][len_topics] = len(articles_of_clusters[x])


    for x in range(0, len_clusters):
        most_topic_val = 0
        most_topic = 0

        for y in range(0, len_topics):
            if confusion_matrix_to_return[x][y] > most_topic_val:
                most_topic = model_topics[y]
                most_topic_val = confusion_matrix_to_return[x][y]
        clusters_and_topics[x] = most_topic

    return confusion_matrix_to_return, clusters_and_topics, articles_of_clusters




def divide_clusters(data):
    clusters = {}
    for article_index in range(0, len(data)):
        selected_cluster = (1+article_index) % NUM_CLUSTERS
        if selected_cluster == 0:
            selected_cluster = NUM_CLUSTERS
            #if not inserted to dict yet, create it
        if selected_cluster not in clusters:
            clusters[selected_cluster] = []
        clusters[selected_cluster].append(article_index)
    return clusters


def get_the_topics_lst(topicstxt):
    """

    :param topicstxt:
    :return:
    """
    list_of_topics = []
    with open(topicstxt) as file:
        for row in file:
            list_of_topics.append(row.strip())
    return list_of_topics

def make_train_set(train_file):
    """

    :param train_file:
    :return:
    """

    """
    We want to use only words that appeared more then 3 times
    """
    headers_all_docs = {}  # holds all articles' headers
    index_of_header = 0  # holds an id that represent a key which connect between headers dic and the articles dic
    index_of_article = 0
    all_articles = {}  # holds the articles
    words_set_count = {}

    with open(train_file) as f:
        for row in f:
            divided_row = row.strip().split(' ')
            if len((divided_row[0].split('\t'))) > 1:
                headers_all_docs[index_of_header] = divided_row[0].replace("<", "").replace(">", "").split("\t")
                index_of_header += 1
            else:  # an article
                doc_content = divided_row
                all_articles[index_of_article] = doc_content
                index_of_article += 1

                for word in doc_content:
                    if word not in words_set_count:
                        words_set_count.setdefault(word, 1)
                    else:
                        words_set_count[word] += 1


    words_with_freqs_more_than_three = delete_rare_words(words_set_count)
    train_aftre_del = delete_rare_words_from_train(words_with_freqs_more_than_three, all_articles)
    train_by_freq = words_count_for_article(train_aftre_del)
    return headers_all_docs, train_aftre_del, words_with_freqs_more_than_three, train_by_freq

def words_count_for_article(articles_train):
    """

    :param articles_train:
    :return:
    """
    count_each_article_words = {}
    for cure_article, words_content in articles_train.iteritems():
        #use Counter method - A Counter is a dict subclass for counting hashable objects.
        count_each_article_words[cure_article] = Counter(words_content)
        #return the dict
    return count_each_article_words


def delete_rare_words_from_train(words_lst_not_rare, train_articles):
    """

    :param words_lst_not_rare:
    :param train_articles:
    :return:
    """
    articles_train_after_del_rare_words = {}
    """
    for each article remove rares words
    """
    for cure_article, content in train_articles.iteritems():
        article_content_after_del_rares = []
        for word in content:
            if word in words_lst_not_rare:
                article_content_after_del_rares.append(word)
        articles_train_after_del_rare_words[cure_article] = article_content_after_del_rares
    return articles_train_after_del_rare_words


def delete_rare_words(list_words_by_frequency):
    """
    delete_rare_words
    Time and place complexity - In order to reduce time and place complexity you should
    filter rare words. A rare word, for this exercise, is a word that occurs 3 times or less in
    the input corpus (develop.txt).
    :param list_words_by_frequency:
    :return:
    """
    #the list to return
    list_words_return = {}
    for word, frequency in list_words_by_frequency.iteritems():
        #return only words with frequency grater than 3
        if 3<frequency:
            list_words_return[word] = frequency

    return list_words_return