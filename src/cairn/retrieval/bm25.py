import math


def idf(n_docs, doc_freq):
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def tf_weight(tf, doc_length, avg_length, k1=1.2, b=0.75):
    length_factor = 1 - b + b * doc_length / avg_length
    return tf * (k1 + 1) / (tf + k1 * length_factor)
