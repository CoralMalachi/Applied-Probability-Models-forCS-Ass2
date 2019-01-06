from __future__ import division
import matplotlib.pyplot as plt

import numpy as np

import math

import help_funcs as ut

def compute_the_likelihood(m_vals, z_vals, k_val):
    """

    :param m_vals:
    :param z_vals:
    :param k_val:
    :return:
    """
    comp_likelihood = 0
    len_of_m_list = len(m_vals)

    for t in range(len_of_m_list):
        sum_zi_e = 0
        zi_length = len(z_vals[t])
        for i in range(0, zi_length):
            curr_zi_m = z_vals[t][i] - m_vals[t]
            if curr_zi_m >= k_val*(-1.0):
                sum_zi_e += math.exp(curr_zi_m)
        comp_likelihood += np.log(sum_zi_e)+m_vals[t]
    return comp_likelihood


def calc_initial_alpha_and_prob(relevant_words_with_freqs, articles_with_their_freq, clusters_of_articles, number_of_clusters,
                                voc_size, lambda_val):

    weights = {}
    # Initialize the weights by the initial clusters - if the doc in the cluster - set the weight to 1, else - set to 0
    for i, doc_list in clusters_of_articles.iteritems():
        for t in doc_list:
            weights[t] = {}
            weights[t][i-1] = 1
            for m in range(0, number_of_clusters):
                if m not in weights[t]:
                    weights[t][m] = 0

    # Initialize the alpha and probs like in m step
    alpha, probabilities = m_step(weights, articles_with_their_freq, relevant_words_with_freqs, number_of_clusters, lambda_val, voc_size)
    return alpha, probabilities


# Calculate perplexity by the given formula of th exercise
def compute_perplexity(lan_likelihood, number_of_words):
    return math.pow(2, (-1 / number_of_words * lan_likelihood))


def em_process(articles_with_their_words_freqs, all_words_with_all_freq, words_clusters, number_of_clusters):
    em_threshold = 10
    k_param = 10
    lambda_val = 1.1

    v_size = len(all_words_with_all_freq)
    # First we will initialize the Pik and Alpha_i for the model
    alpha, probabilities = calc_initial_alpha_and_prob(all_words_with_all_freq, articles_with_their_words_freqs,
                                                       words_clusters, number_of_clusters, v_size,
                                                       lambda_val)

    likelihood_array = []
    perplexity_array = []
    # Initial value for the algorithm to continue running
    prev_likelihood = -10000101
    curr_likelihood = -10000000
    epoch = 0
    number_of_words = sum(all_words_with_all_freq.values())
    # The em will continue running until the current calculated likelihood is smaller from the previous calculated
    # likelihood
    while curr_likelihood - prev_likelihood > em_threshold:
        # In the e-step the algorithm calculates the weights of each document to be in a cluster
        # And returns them and the list of z and m (for the likelihood)
        w, z_list, m_list = start_e_step(articles_with_their_words_freqs, alpha, probabilities,number_of_clusters, k_param)
        # In the m-step the algorithm calculates the alphas and probs according to the givem weight values
        alpha, probabilities = m_step(w, articles_with_their_words_freqs, all_words_with_all_freq, number_of_clusters,
                                      lambda_val, v_size)
        prev_likelihood = curr_likelihood
        # Calc the lan likelihood of the model
        curr_likelihood = compute_the_likelihood(m_list, z_list, k_param)
        print curr_likelihood
        # Calc the model's perplexity
        curr_perplexity = compute_perplexity(curr_likelihood, number_of_words)

        likelihood_array.append(curr_likelihood)
        perplexity_array.append(curr_perplexity)
        epoch += 1

    # Create the graphs of the likelihood and perplexity per epoch
    ploting_results_into_graph(epoch, likelihood_array, "likelihood")
    ploting_results_into_graph(epoch, perplexity_array, "perplexity")

    # Return the final weights
    return w


def ploting_results_into_graph(epochs, axis_y, label):
    """

    :param epochs:
    :param axis_y:
    :param label:
    :return:
    """
    axis_x = [i for i in range(0, epochs)]  # number of iterations
    plt.plot(axis_x, axis_y, label=label)
    plt.xlabel("epochs")
    plt.ylabel(label)
    #plt.title("I vs L Graph")
    plt.xlim(0, epochs)
    plt.ylim(min(axis_y), max(axis_y))
    plt.legend(loc="lower right")
    plt.savefig(label + "2.png")


def start_e_step(articles_with_their_words_freqs, alpha_val, probs, clusters_len, k_val):
    w = {}
    z_list = []
    m_list = []
    # For every document in our training set we want to calculate it's possibility to be in the clusters
    for t, doc_with_freq in articles_with_their_words_freqs.iteritems():
        w[t] = {}
        # Calculate the z array (for every cluster there is it's own value of z) for the current doc
        curr_z, max_zi = calc_z_values( clusters_len, alpha_val, probs, doc_with_freq)
        sum_zi = 0
        for i in range(0, clusters_len):
            if curr_z[i] - max_zi < (-1.0) * k_val:
                w[t][i] = 0
            else:
                w[t][i] = math.exp(curr_z[i] - max_zi)
                sum_zi += w[t][i]
        for i in range(0, clusters_len):
            w[t][i] /= sum_zi

        z_list.append(curr_z)
        m_list.append(max_zi)
    return w, z_list, m_list


# Calculate the z array for a current doc by the equation
def calc_z_values( number_of_clusters, alpha, probabilities, curr_article_with_t):
    z = []
    for i in range(0, number_of_clusters):
        sum_of_freq_ln = 0
        for word in curr_article_with_t:
            sum_of_freq_ln += curr_article_with_t[word] * np.log(probabilities[word][i])
        z.append(np.log(alpha[i]) + sum_of_freq_ln)
    max_z = max(z)
    return z, max_z


def m_step(weights, articles_with_their_words_frequencies, relevant_words_with_freq, number_of_clusters, lambda_val, v_size):
    print('m_step')
    threshold = 0.000001
    number_of_docs = len(articles_with_their_words_frequencies)
    probabilities = {}
    denominator = []
    # For every cluster we want to calculate the new probs for words to be in the clusters
    # The calculation is by the wti (the probability of the doc to be in the cluster)
    for i in range(0, number_of_clusters):
        denom_i = 0
        for t in articles_with_their_words_frequencies:
            len_of_t = sum(articles_with_their_words_frequencies[t].values())
            denom_i += weights[t][i] * len_of_t
        denominator.append(denom_i)
    for word in relevant_words_with_freq:
        probabilities[word] = {}
        for i in range(0, number_of_clusters):
            numerator = 0
            for t in articles_with_their_words_frequencies:
                if word in articles_with_their_words_frequencies[t] and weights[t][i] != 0:
                    numerator += weights[t][i] * articles_with_their_words_frequencies[t][word]
            probabilities[word][i] = ut.calc_lidstone(numerator, denominator[i], v_size, lambda_val)

    # If alpha is smaller then a threshold we will scale it to the threshold to not get ln(alpha) = error

    alpha = [0] * number_of_clusters
    for i in range(0, number_of_clusters):
        for t in articles_with_their_words_frequencies:
            alpha[i] += weights[t][i]
        alpha[i] /= number_of_docs
    # alpha = [sum(i) / number_of_docs for i in zip(*weights)]
    for i in range(0, len(alpha)):
        if alpha[i] < threshold:
            alpha[i] = threshold
    sum_of_alpha = sum(alpha)
    # Normalize alpha for it to sum to 1
    alpha = [x / sum_of_alpha for x in alpha]
    return alpha, probabilities
