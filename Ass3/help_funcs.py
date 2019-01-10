from __future__ import division


import numpy as np

from collections import Counter

NUM_CLUSTERS = 9

# Calculate the accuracy of the model
def calc_accuracy(articles_headers, all_doc_with_classification):
    accuracy = 0
    for doc in all_doc_with_classification:
        if doc[1] in articles_headers[doc[0]]:
            accuracy += 1

    total_articles = len(articles_headers)
    return accuracy / total_articles

def assign_classifications_to_docs(clusters_with_topics,documents_in_clusters):
    docs_with_assignments = []
    for index_of_cluster in documents_in_clusters:
        for t in documents_in_clusters[index_of_cluster]:
            classification = clusters_with_topics[index_of_cluster]
            docs_with_assignments.append((t, classification))
    return docs_with_assignments

# Will be helpful in smoothing
def calc_lidstone_for_unigram(count_words, train_size, voc_size, m_lambda):
    # C(X)+ LAMBDA / |S| + LAMBDA*|X|
    return (count_words + m_lambda) / (train_size + m_lambda * voc_size)


def create_confusion_matrix(weights, articles_with_their_words_freqs,topics_list, article_topics):
    documents_in_clusters = {}

    number_of_topics = len(topics_list)
    number_of_clusters = len(weights[0].keys())

    # Set each cluster with it's corresponding cluster by wti
    for t in articles_with_their_words_freqs:
        max_weights = weights[t][0]
        selected_index = 0
        for i in range(0, number_of_clusters):
            if weights[t][i] > max_weights:
                max_weights = weights[t][i]
                selected_index = i
        if selected_index not in documents_in_clusters:
            documents_in_clusters[selected_index] = []
        documents_in_clusters[selected_index].append(t)

    # Create the confusion matrix
    conf_matrix = np.zeros((number_of_clusters, number_of_topics + 1))
    for row in range(0, number_of_clusters):
        for col in range(0, number_of_topics):
            current_topic = topics_list[col]
            for t in documents_in_clusters[row]:
                if current_topic in article_topics[t]:
                    conf_matrix[row][col] += 1
        # Number of articles in the cluster
        conf_matrix[row][number_of_topics] = len(documents_in_clusters[row])

    clusters_with_topics = {}
    for row in range(0, number_of_clusters):
        dominant_topic = 0
        dominant_topic_val = 0
        for col in range(0, number_of_topics):
            if conf_matrix[row][col] > dominant_topic_val:
                dominant_topic = topics_list[col]
                dominant_topic_val = conf_matrix[row][col]
        clusters_with_topics[row] = dominant_topic

    return conf_matrix, clusters_with_topics, documents_in_clusters

def split_into_clusters(data):
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


def get_topics(topicstxt):
    """

    :param topicstxt:
    :return:
    """
    list_of_topics = []
    with open(topicstxt) as file:
        for row in file:
            list_of_topics.append(row.strip())
    return list_of_topics

def create_train_data(train_file):
    """

    :param train_file:
    :return:
    """
    headers_train_data = {}  # holds all articles' headers
    header_id = 0  # holds an id that represent a key which connect between headers dic and the articles dic
    article_id = 0
    articles_train_data = {}  # holds the articles
    all_words_with_counter = {}

    with open(train_file) as f:
        for line in f:
            splited_line = line.strip().split(' ')
            if len((splited_line[0].split('\t'))) > 1:  # note header
                headers_train_data[header_id] = splited_line[0].replace("<", "").replace(">", "").split("\t")
                header_id += 1
            else:  # an article
                article_content = splited_line
                articles_train_data[article_id] = article_content
                article_id += 1

                for word in article_content:
                    if word not in all_words_with_counter:
                        all_words_with_counter.setdefault(word, 1)
                    else:
                        all_words_with_counter[word] += 1

    # We want to use only words that appeared more then 3 times
    relevant_words_with_freqs = clean_rare_words(all_words_with_counter)
    relevant_articles_train_data = clean_rare_words_train_data(relevant_words_with_freqs, articles_train_data)
    article_train_data_with_freq = get_words_freq_for_article(relevant_articles_train_data)
    return headers_train_data, relevant_articles_train_data, relevant_words_with_freqs, article_train_data_with_freq

def get_words_freq_for_article(articles_train):
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


def clean_rare_words_train_data(words_lst_not_rare, train_articles):
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


def clean_rare_words(list_words_by_frequency):
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