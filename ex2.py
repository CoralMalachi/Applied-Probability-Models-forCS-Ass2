'''
Coral Malachi	314882853
Avishai Zagoury 209573633
'''
import sys
import math

VOCABULARY_SIZE = 300000


def word_count(words_dict, input_word):
    """
    Returns and number of appearances of given word
    """
    if input_word in words_dict:
        return words_dict[input_word]
    return 0


def calc_lidstone(words_dict, input_word, corpus_size, lamda):
    """
    Calculates lidstone smoothing of MLE according to given lamda
    """

    appearances = word_count(words_dict, input_word)
    num_of_words = len(words_dict)
    return float(appearances + lamda) / (corpus_size + lamda * num_of_words)


def calc_lidstone_perplexity(text_words, words_dict, corpus_size, lamda):
    log_sum = 0
    for word in text_words:
        word_lidstone_rank = calc_lidstone(words_dict, word, corpus_size, lamda)
        log_sum += math.log(word_lidstone_rank, 2)
    return 2 ** ((-1. / len(text_words)) * log_sum)


def find_best_lamda(text_words, words_dict, corpus_size, lamda_interval):
    """
    Try all lamdas in interval and find the lamda that minimizes the preplexity
    """

    min_preplexity = 10000000  # Big number
    best_lamda = lamda_interval[0]
    for lamda in lamda_interval:
        print lamda
        preplexity = calc_lidstone_perplexity(text_words, words_dict, corpus_size, lamda)
        if preplexity < min_preplexity:
            min_preplexity = preplexity
            best_lamda = lamda

    return best_lamda, min_preplexity


def main(dev_file, test_file, input_word, output_file):
    output_fd = open(output_file, "w")
    output_fd.write("#Students\tCoral_Malachi_Avishai_Zagoury\t<314882853>_209573633\n")
    output_fd.write("#Output1\t{}\n".format(dev_file))
    output_fd.write("#Output2\t{}\n".format(test_file))
    output_fd.write("#Output3\t{}\n".format(input_word))
    output_fd.write("#Output4\t{}\n".format(output_file))
    output_fd.write("#Output5\t{}\n".format(VOCABULARY_SIZE))
    output_fd.write("#Output6\t{}\n".format(1. / VOCABULARY_SIZE))

    with open(dev_file, "r") as fd_r:
        dev_data = fd_r.read()
    dev_data = dev_data.split('\n')[1::2]
    dev_data = " ".join(dev_data).split()

    dev_size = len(dev_data)
    output_fd.write("#Output7\t{}\n".format(dev_size))

    val_data = dev_data[int(dev_size * 0.9):]
    train_data = dev_data[:int(dev_size * 0.9)]
    train_size = len(train_data)
    output_fd.write("#Output8\t{}\n".format(len(val_data)))
    output_fd.write("#Output9\t{}\n".format(train_size))

    # This dict counts appearence of every known word
    train_words_count = dict()
    for word in train_data:
        if word not in train_words_count:
            train_words_count[word] = 1
        else:
            train_words_count[word] += 1

    known_words_size = len(train_words_count)
    output_fd.write("#Output10\t{}\n".format(known_words_size))
    output_fd.write("#Output11\t{}\n".format(word_count(train_words_count, input_word)))
    # Calculate MLE of a word - 0 if the word wasn't observed
    output_fd.write("#Output12\t{}\n".format(float(word_count(train_words_count, input_word)) / train_size))
    # Calculate MLE of an unseen word
    output_fd.write("#Output13\t{}\n".format(float(word_count(train_words_count, "unseen-word")) / train_size))
    word_lidstone = calc_lidstone(train_words_count, input_word, train_size, 0.1)
    output_fd.write("#Output14\t{}\n".format(word_lidstone))
    unseen_word_lidstone = calc_lidstone(train_words_count, "unseen-word", train_size, 0.1)
    output_fd.write("#Output15\t{}\n".format(unseen_word_lidstone))
    output_fd.write("#Output16\t{}\n".format(calc_lidstone_perplexity(val_data, train_words_count, train_size, 0.01)))
    output_fd.write("#Output17\t{}\n".format(calc_lidstone_perplexity(val_data, train_words_count, train_size, 0.1)))
    output_fd.write("#Output18\t{}\n".format(calc_lidstone_perplexity(val_data, train_words_count, train_size, 1.)))

    # Calc best lamda
    lamda_interval = [0.01 + x * 0.01 for x in xrange(200)]
    best_lamda, best_preplexity = find_best_lamda(val_data, train_words_count, train_size, lamda_interval)
    output_fd.write("#Output19\t{}\n".format(best_lamda))
    output_fd.write("#Output20\t{}\n".format(best_preplexity))

    # TODO: Output 21-24 koral should do

    with open(test_file, "r") as fd_r:
        test_data = fd_r.read()
    test_data = test_data.split('\n')[1::2]
    test_data = " ".join(test_data).split()

    output_fd.write("#Output25\t{}\n".format(len(test_data)))
    test_lidstone_preplexity = calc_lidstone_perplexity(test_data, train_words_count, train_size, best_lamda)
    output_fd.write("#Output26\t{}\n".format(test_lidstone_preplexity))

    # TODO: Output 27,29 koral should do
    test_heldout_preplexity = 0  # TODO: fill this
    output_fd.write("#Output28\t{}\n".format('L' if test_lidstone_preplexity < test_heldout_preplexity else 'H'))


if __name__ == "__main__":
    main(*sys.argv[1:5])
