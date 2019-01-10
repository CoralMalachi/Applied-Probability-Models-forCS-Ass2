'''
Coral Malachi	314882853
Avishai Zagoury 209573633
'''
from __future__ import division
# import matplotlib.pyplot as plt

import numpy as np

import math

import help_funcs as ut
MODEL_THRESHOLD = 0.000001
EM_THRESHOLD = 10
K_PARAM = 10
LAMBDA_VAL = 1.1

def plot_graph(epochs, axis_y, label):
    """
    :param epochs:
    :param axis_y:
    :param label:
    :return:
    """
    axis_x = [i for i in range(0, epochs)]

    plt.plot(axis_x, axis_y, label=label)
    plt.xlabel("epochs")
    plt.ylabel(label)
    plt.xlim(0, epochs)
    plt.ylim(min(axis_y), max(axis_y))
    plt.legend(loc="lower right")
    plt.savefig(label + "2.png")



def compute_the_likelihood(ms, zs, value_of_k):
    """

    :param ms:
    :param zs:
    :param value_of_k:
    :return:likelihood
    """


    cure_likelihood = 0
    for t in range(len(ms)):
        sum_of_zs_e = 0
        for i in range(0, len(zs[t])):
            curr_zi_m = zs[t][i] - ms[t]
            if curr_zi_m >= (-1.0) * value_of_k:
                sum_of_zs_e += math.exp(curr_zi_m)
        log_sum = np.log(sum_of_zs_e)
        cure_likelihood += log_sum + ms[t]
    return cure_likelihood






def compute_perp(lan_likelihood, words_set_length):
    return math.pow(2, (-1 / words_set_length * lan_likelihood))


def run_em_algorithm(articles_and_freqs, words_set, clusters_of_words, clusters_length):
    """

    :param articles_and_freqs:
    :param words_set:
    :param clusters_of_words:
    :param clusters_length:
    :return:final weights
    """

    """
        according to Underflow Scaling and Smoothing in EM article,
        The em algorithm continue running as long as the current calculated likelihood is bigger than the previous likelihood
    """
    likelihood_lst = []
    iter_index = 0
    #init likelihood vals
    current_val_of_likelihood = -10000000
    previous_val_of_likelihood = -10000101


    perplexity_lst = []
    value_k = K_PARAM
    em_thresh = EM_THRESHOLD
    lambda_val = LAMBDA_VAL

    v_size = len(words_set)
    alpha, probabilities = init_probs_and_alfa(words_set, articles_and_freqs,
                                                       clusters_of_words, clusters_length, v_size,
                                                       lambda_val)


    words_set_length = sum(words_set.values())

    while current_val_of_likelihood - previous_val_of_likelihood > em_thresh:

        w, zs, ms = EM_algorithm_e_step(articles_and_freqs, alpha, probabilities,clusters_length, value_k)

        alpha, probabilities = EM_algorithm_m_step(w, articles_and_freqs, words_set, clusters_length,
                                      lambda_val, v_size)
        previous_val_of_likelihood = current_val_of_likelihood

        current_val_of_likelihood = compute_the_likelihood(ms, zs, value_k)
        print current_val_of_likelihood

        compute_current_perplexity = compute_perp(current_val_of_likelihood, words_set_length)

        likelihood_lst.append(current_val_of_likelihood)
        perplexity_lst.append(compute_current_perplexity)
        iter_index += 1


    # plot_graph(iter_index, likelihood_lst, "likelihood")
    # plot_graph(iter_index, perplexity_lst, "perplexity")


    return w


def init_probs_and_alfa(words_set, articles, docs_clusters, clusters_length,
                                vocab_len, value_of_lambda):
    """

    :param words_set:
    :param articles:
    :param docs_clusters:
    :param clusters_length:
    :param vocab_len:
    :param value_of_lambda:
    :return:
    """
    """
     according to Underflow Scaling and Smoothing in EM article,we need
      to init the weights by the initial clusters
      it means that if the article in the cluster  
      the weight is equal to 1,
      else the weight is equal to 0
      in addition we need to init  alpha and probs like in m step
    """
    #init
    lst_of_weights = {}

    for i, lst_of_articles in docs_clusters.iteritems():
        for article_cure in lst_of_articles:
            lst_of_weights[article_cure] = {}
            lst_of_weights[article_cure][i-1] = 1
            for x in range(0, clusters_length):
                if x not in lst_of_weights[article_cure]:
                    lst_of_weights[article_cure][x] = 0


    # alfa, probs = EM_algorithm_m_step(lst_of_weights, articles, words_set, clusters_length, value_of_lambda, vocab_len)
    return EM_algorithm_m_step(lst_of_weights, articles, words_set, clusters_length, value_of_lambda, vocab_len)


def EM_algorithm_e_step(docs, value_of_alpha, probs, clusters_length, value_of_k):
    """
    :param words_set:
    :param docs:
    :param value_of_alpha:
    :param probs:
    :param clusters_length:
    :param value_of_k:
    :return:w, z_values, m_vals
    """
    """
    Define zi to be the log of the numerator of wti  
    Now we have various e^zi which are unstable. To solve this we define: m = maxi(zi)
    Function Action:
    1.  For each doc in our train data we are going  to compute
        the possibility to belong to clusters
    2.  then we compute the z array for the current document
    """
    z_values = []
    w = {}

    m_vals = []

    for t, doc_with_freq in docs.iteritems():
        w[t] = {}
        z_value_current_sum = 0
        z_value, max_zi = compute_zs(clusters_length, value_of_alpha, probs, doc_with_freq)

        for i in range(0, clusters_length):
            sub = z_value[i] - max_zi
            """
            according to Underflow Scaling and Smoothing in EM article,
             if sub value is less than -k we should set w value to zero
            """
            if sub < value_of_k *(-1.0):
                w[t][i] = 0
            else:
                w[t][i] = math.exp(sub)
                z_value_current_sum += w[t][i]

        for x in range(0, clusters_length):
            w[t][x] /= z_value_current_sum

        #update the 2 lists with the new values
        z_values.append(z_value)
        m_vals.append(max_zi)
    return w, z_values, m_vals


def compute_zs(clusters_length, value_of_alpha, probs, curr_doc):
    """

    :param clusters_length:
    :param value_of_alpha:
    :param probs:
    :param curr_doc:
    :return:
    """
    """
    function action : according to Underflow Scaling and Smoothing in EM article,
    we need to compute the z array for a current article by the equation
    """
    z_cure = []
    for x in range(0, clusters_length):
        sum_ln = 0
        for word in curr_doc:
            log_val = np.log(probs[word][x])
            sum_ln += log_val * curr_doc[word]
        z_to_add=np.log(value_of_alpha[x]) + sum_ln
        z_cure.append(z_to_add)
    # max_z = max(z_cure)
    return z_cure, max(z_cure)


def EM_algorithm_m_step(model_weights, articles, words_set, clusters_length, value_of_lambda, vocab_len):
    """

    :param model_weights:
    :param articles:
    :param words_set:
    :param clusters_length:
    :param value_of_lambda:
    :param vocab_len:
    :return:
    """
    """
    function action : according to Underflow Scaling and Smoothing in EM article,
    we need to compute the new probs for words to be in the clusters For each cluster 
    in addition we normalize alpha to make sure its sum is 1
    """

    number_articles = len(articles)
    alpha_val = [0] * clusters_length
    probs = {}
    model_threshold = MODEL_THRESHOLD
    denominate_lst = []
    for x in range(0, clusters_length):
        cure_den = 0
        for m_article in articles:
            cure_den += model_weights[m_article][x] * sum(articles[m_article].values())
        denominate_lst.append(cure_den)
    for word in words_set:
        probs[word] = {}
        for y in range(0, clusters_length):
            m_numerator = 0
            for t in articles:
                if word in articles[t] and model_weights[t][y] != 0:
                    m_numerator += model_weights[t][y] * articles[t][word]
            probs[word][y] = ut.compute_lidstone(m_numerator, denominate_lst[y], vocab_len, value_of_lambda)


    for i in range(0, clusters_length):
        for t in articles:
            alpha_val[i] += model_weights[t][i]
        alpha_val[i] /= number_articles

    for i in range(0, len(alpha_val)):
        if alpha_val[i] < model_threshold:
            alpha_val[i] = model_threshold
    alfa_sum = sum(alpha_val)

    alpha_val = [x / alfa_sum for x in alpha_val]
    return alpha_val, probs
