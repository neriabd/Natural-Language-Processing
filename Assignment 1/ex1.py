import spacy
from datasets import load_dataset
from collections import defaultdict
import math


def main():
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    unigram_dict, length_doc = train_unigram(text, nlp)
    bigram_dict = train_bigram(text, nlp)
    print('q2:', complete_sentence_bigram(nlp("I have a house in"), bigram_dict))
    sentence1 = nlp("Brad Pitt was born in Oklahoma")
    sentence2 = nlp("The actor was born in USA")
    sentence1_p = bigram_sentence(bigram_dict, sentence1)
    sentence2_p = bigram_sentence(bigram_dict, sentence2)
    print('q3a sentence 1: ', sentence1_p)
    print('q3a sentence 2: ', sentence2_p)
    print('q3b: ', perplexity([sentence1, sentence2], [sentence1_p, sentence2_p]))
    sentence1_smooth_p = smoothing_probability(1 / 3, 2 / 3, unigram_dict, bigram_dict, sentence1, length_doc)
    sentence2_smooth_p = smoothing_probability(1 / 3, 2 / 3, unigram_dict, bigram_dict, sentence2, length_doc)
    print('q4 probability sentence 1: ', sentence1_smooth_p)
    print('q4 probability sentence 2: ', sentence2_smooth_p)
    print('q4 perplexity: ', perplexity([sentence1, sentence2], [sentence1_smooth_p, sentence2_smooth_p]))


def calc_sum_log(probabilities):
    sum_log = 0
    for probability in probabilities:
        if probability != 0:
            sum_log += probability

    return sum_log


def calc_M(sentences):
    M = 0
    for sentence in sentences:
        M += len(sentence)

    return M


def perplexity(sentences, sentences_probabilities):
    M = calc_M(sentences)
    sum_log = calc_sum_log(sentences_probabilities)
    l = sum_log / M
    return math.exp(-l)


def unigram_sentence(unigram_dict, sentence, length_doc):
    p = 0
    for word in sentence:
        if word.lemma_ not in unigram_dict:
            return float('-inf')
        p += math.log(unigram_dict[word.lemma_] / length_doc)

    return p


def train_unigram(text, nlp):
    unigram_dict = dict()
    counter = 0
    for t in text:
        line = nlp(t['text'])
        for word in line:
            if word.is_alpha:
                counter += 1
                unigram_dict[word.lemma_] = unigram_dict.get(word.lemma_, 0) + 1

    return unigram_dict, counter


def bigram_sentence(bigram_dict, sentence):
    p = 0
    prev_word = 'START'
    for word in sentence:
        if prev_word not in bigram_dict or word.lemma_ not in bigram_dict[prev_word]:
            return float('-inf')

        p += math.log(bigram_dict[prev_word][word.lemma_] / sum(bigram_dict[prev_word].values()))
        prev_word = word.lemma_

    return p


def train_bigram(text, nlp):
    bigram_dict = defaultdict(dict)
    for t in text:
        prev_word = 'START'
        line = nlp(t['text'])
        for word in line:
            if word.is_alpha:
                bigram_dict[prev_word][word.lemma_] = bigram_dict[prev_word].get(word.lemma_, 0) + 1
                prev_word = word.lemma_

    return bigram_dict


def complete_sentence_bigram(sentence, bigram_dict):
    last_word = sentence[-1]
    if last_word.lemma_ not in bigram_dict:
        return

    return max(bigram_dict[last_word.lemma_], key=bigram_dict[last_word.lemma_].get)


def smoothing_probability(lambda1, lambda2, unigram_dict, bigram_dict, sentence, length_doc):
    p = 0
    prev_word = 'START'
    for word in sentence:
        word_p = 0
        if word.lemma_ in unigram_dict:
            word_p += (unigram_dict[word.lemma_] / length_doc) * lambda1
        if prev_word in bigram_dict and word.lemma_ in bigram_dict[prev_word]:
            word_p += bigram_dict[prev_word][word.lemma_] / sum(bigram_dict[prev_word].values()) * lambda2
        prev_word = word.lemma_

        p += math.log(word_p)
    return p


if __name__ == '__main__':
    main()
