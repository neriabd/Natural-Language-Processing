import re
import nltk
from nltk.corpus import brown
import ssl
from collections import defaultdict
import numpy as np
import pandas as pd


def main():
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context

    nltk.download('brown')
    sentences_array = brown.tagged_sents(categories='news')
    split_index = int(0.9 * len(sentences_array))
    train_set = sentences_array[:split_index]
    test_set = sentences_array[split_index:]
    word_dict, tag_dict, tag_bigram_dict, name_lst = train(train_set)
    row_names = list(tag_bigram_dict.keys())

    # b
    accuracy_known, accuracy_unknown, accuracy_total = error_rate(test_set, word_dict)
    print_errors(" ", accuracy_known, accuracy_unknown, accuracy_total)

    num_words, freq_total = calc_unique_words(train_set, test_set)

    # c
    emissions_word_tag_dict, emissions_tag_word_dict = emissions(tag_dict)
    transitions_dict = transitions(tag_bigram_dict)
    all_tags = bigram_viterbi(test_set, emissions_word_tag_dict, emissions_tag_word_dict, transitions_dict,
                              row_names, False, num_words, freq_total)
    accuracy_known, accuracy_unknown, accuracy_total = error_rate_viterbi(all_tags, test_set, word_dict)
    print_errors(" viterbi ", accuracy_known, accuracy_unknown, accuracy_total)

    # d
    emissions_word_tag_dict_add_1, emissions_tag_word_dict_add_1 = emissions_add_one(tag_dict, num_words, freq_total)
    all_tags2 = bigram_viterbi(test_set, emissions_word_tag_dict_add_1, emissions_tag_word_dict_add_1, transitions_dict,
                               row_names, True, num_words, freq_total)

    accuracy_known, accuracy_unknown, accuracy_total = error_rate_viterbi(all_tags2, test_set, word_dict)
    print_errors(" viterbi add one ", accuracy_known, accuracy_unknown, accuracy_total)

    # e1
    pseudo_word_dict, pseudo_tag_dict = train_pseudo(word_dict, tag_dict, name_lst)
    emissions_word_tag_dict, emissions_tag_word_dict = emissions(pseudo_tag_dict)
    test_set = update_test(test_set, pseudo_word_dict, name_lst)
    all_tags = bigram_viterbi(test_set, emissions_word_tag_dict, emissions_tag_word_dict, transitions_dict,
                              row_names, False, num_words, freq_total)
    accuracy_known, accuracy_unknown, accuracy_total = error_rate_viterbi(all_tags, test_set, pseudo_word_dict)
    print_errors(" viterbi pseudo ", accuracy_known, accuracy_unknown, accuracy_total)

    # e2
    emissions_word_tag_dict, emissions_tag_word_dict = emissions_add_one(pseudo_tag_dict, len(pseudo_word_dict),
                                                                         freq_total)
    all_tags = bigram_viterbi(test_set, emissions_word_tag_dict, emissions_tag_word_dict, transitions_dict,
                              row_names, True, num_words, freq_total)
    accuracy_known, accuracy_unknown, accuracy_total = error_rate_viterbi(all_tags, test_set, pseudo_word_dict)
    print_errors(" viterbi add one pseudo ", accuracy_known, accuracy_unknown, accuracy_total)

    confusion_matrix = build_confusion_matrix(test_set, len(tag_bigram_dict.keys()), all_tags, row_names,
                                              pseudo_tag_dict)
    print(confusion_matrix)



def print_errors(question_name, accuracy_known, accuracy_unknown, accuracy_total):
    error_known = 1 - accuracy_known
    error_unknown = 1 - accuracy_unknown
    error_total = 1 - accuracy_total
    print("Error rate for known words" + question_name + "is", round(error_known, 5))
    print("Error rate for unknown words" + question_name + "is", round(error_unknown, 5))
    print("Total error rate" + question_name + "is", round(error_total, 5))
    print()


def train(sentences_array):
    word_dict = defaultdict(dict)
    tag_dict = defaultdict(dict)
    tag_bigram_dict = defaultdict(dict)
    name_lst = set()
    for line in sentences_array:
        prev_tag = 'START'
        counter = 0
        for word, tag in line:
            if counter != 0 and word[0].isupper():
                name_lst.add(word)
            new_tag = simplify_tag(tag)
            word_dict[word][new_tag] = word_dict[word].get(new_tag, 0) + 1
            tag_dict[new_tag][word] = tag_dict[new_tag].get(word, 0) + 1
            tag_bigram_dict[prev_tag][new_tag] = tag_bigram_dict[prev_tag].get(new_tag, 0) + 1
            prev_tag = new_tag
            counter += 1
        tag_bigram_dict[prev_tag]["STOP"] = tag_bigram_dict[prev_tag].get("STOP", 0) + 1

    return word_dict, tag_dict, tag_bigram_dict, name_lst


def train_pseudo(word_dict, tag_dict, name_lst):
    pseudo_word_dict = defaultdict(dict)
    pseudo_tag_dict = defaultdict(dict)
    for word in word_dict:
        if sum(word_dict[word].values()) >= 3:
            pseudo_word_dict[word] = word_dict[word]
            for tag in word_dict[word]:
                pseudo_tag_dict[tag][word] = tag_dict[tag][word]
        else:
            new_word = classify_word(word, name_lst)
            for tag in word_dict[word]:
                pseudo_word_dict[new_word][tag] = pseudo_word_dict[new_word].get(tag, 0) + word_dict[word][tag]
                pseudo_tag_dict[tag][new_word] = pseudo_tag_dict[tag].get(new_word, 0) + tag_dict[tag][word]

    return pseudo_word_dict, pseudo_tag_dict


def classify_word(word, name_lst):
    if word.isdigit():
        return "NUMBER"
    if word in name_lst:
        return "NAME"
    if word.endswith("ing") or word.endswith("ed"):
        return "VERB"
    if word.endswith("ly"):
        return "ADVERB"
    if word.startswith("$"):
        return "MONEY"
    if bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', word)):
        return "FLOAT"
    if bool(re.match(r'^[+-]?\d{1,3}(,\d{3})*$', word)):
        return "NUMBER"
    if bool(re.match(r'^\d+-\d+$', word)):
        return "RANGE"
    if "-" in word:
        return "PART"
    if word.endswith("'s") or word.endswith("s'"):
        return "SINGLE_QUOTE"
    if bool(re.match(r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b', word)):
        return "TIME"
    if word.endswith("ist"):
        return "IST"
    if word.endswith("ists"):
        return "ISTS"
    if word.endswith("%"):
        return "PERCENT"
    if word.endswith("ion"):
        return "ABBREVIATION"
    if word.endswith("ions"):
        return "ABBREVIATIONS"
    if word.endswith("ment"):
        return "MENT"
    if word.endswith("ments"):
        return "MENTS"
    if word.endswith("ent"):
        return "ENT"

    return "OTHER"


def simplify_tag(new_tag):
    if "$" in new_tag:
        new_tag = new_tag.split('$')[0]
    if "*" in new_tag and new_tag != "*":
        new_tag = new_tag.split('*')[0]
    if "+" in new_tag:
        new_tag = new_tag.split('+')[0]
    if "-" in new_tag and new_tag != "--":
        new_tag = new_tag.split('-')[0]
    return new_tag


def error_rate(sentences_array, train_dict):
    correct_tag_known = 0
    correct_tag_unknown = 0
    counter_known = 0
    counter_unknown = 0
    for line in sentences_array:
        for word, tag in line:
            new_tag = simplify_tag(tag)
            if word not in train_dict:
                counter_unknown += 1
                if "NN" == new_tag:
                    correct_tag_unknown += 1
            else:
                counter_known += 1
                train_tag = max(train_dict[word], key=train_dict[word].get)
                if train_tag == new_tag:
                    correct_tag_known += 1

    accuracy_known = correct_tag_known / counter_known
    accuracy_unknown = correct_tag_unknown / counter_unknown
    accuracy_total = (correct_tag_unknown + correct_tag_known) / (counter_known + counter_unknown)

    return accuracy_known, accuracy_unknown, accuracy_total


def emissions(tag_dict):
    emissions_word_tag = defaultdict(dict)
    emissions_tag_word = defaultdict(dict)
    for tag in tag_dict:
        for word in tag_dict[tag]:
            result = tag_dict[tag][word] / sum(tag_dict[tag].values())
            emissions_word_tag[word][tag] = result
            emissions_tag_word[tag][word] = result

    return emissions_word_tag, emissions_tag_word


def emissions_add_one(tag_dict, num_words, freq_total):
    emissions_word_tag = defaultdict(dict)
    emissions_tag_word = defaultdict(dict)
    for tag in tag_dict:
        for word in tag_dict[tag]:
            result = (tag_dict[tag][word] + 1) / (num_words + freq_total)
            emissions_word_tag[word][tag] = result
            emissions_tag_word[tag][word] = result

    return emissions_word_tag, emissions_tag_word


def transitions(tag_bigram_dict):
    transitions = defaultdict(dict)
    for prev_tag in tag_bigram_dict:
        for tag in tag_bigram_dict[prev_tag]:
            transitions[tag][prev_tag] = tag_bigram_dict[prev_tag][tag] / sum(tag_bigram_dict[prev_tag].values())

    return transitions


def init_tables(row_names, emissions_tag_word_dict, line):
    table = np.zeros((len(emissions_tag_word_dict) + 1, len(line) + 1))
    backpointer = np.zeros((len(emissions_tag_word_dict) + 1, len(line) + 1))
    # backpointer = [['NN' for _ in range(len(line) + 1)] for _ in range(len(emissions_tag_word_dict) + 1)]
    table[row_names.index("START"), 0] = 1
    return table, backpointer


def update_test(test_set, word_dict, name_lst):
    new_test_set = []
    for i in range(len(test_set)):
        line = []
        for j in range(len(test_set[i])):
            word, tag = test_set[i][j]
            if word not in word_dict:
                line.append((classify_word(word, name_lst), tag))
            else:
                line.append(test_set[i][j])
        new_test_set.append(line)

    return new_test_set


def bigram_viterbi(test_set, emissions_word_tag_dict, emissions_tag_word_dict, transitions_dict, row_names,
                   is_add, num_words, freq_total):
    all_tags = []
    counter = 0
    for line in test_set:

        table, backpointer = init_tables(row_names, emissions_tag_word_dict, line)
        for k in range(1, len(line) + 1):
            word, _ = line[k - 1]
            for v in range(len(row_names)):
                tag = row_names[v]

                if word not in emissions_word_tag_dict:
                    if tag == "NN":
                        emissions_word_tag_dict[word][tag] = 1
                    else:
                        continue

                if is_add and word not in emissions_tag_word_dict[tag]:
                    emissions_word_tag_dict[word][tag] = 1 / (freq_total + num_words)

                if tag not in emissions_word_tag_dict[word]:
                    continue

                for u, prev_tag in enumerate(row_names):
                    if prev_tag in transitions_dict[tag]:
                        value = table[u, k - 1] * transitions_dict[tag][prev_tag] * emissions_word_tag_dict[word][tag]

                        if value > table[v, k]:
                            table[v, k] = value
                            backpointer[v, k] = u

        all_tags.append(extract_tags(backpointer, row_names, table, transitions_dict))
        counter += 1

    return all_tags


def extract_tags(backpointer, row_names, table, transitions_dict):
    tags = []
    yk = row_names.index("NN")
    max_value = 0
    for i in range(len(table) - 2):
        tag = row_names[i]
        if tag in transitions_dict["STOP"]:
            value = transitions_dict["STOP"][tag] * table[i, -1]
            if value > max_value:
                max_value = value
                yk = i

    for i in range(table.shape[1] - 1, 0, -1):
        yk_sub_1 = backpointer[np.int64(yk), i]
        string = row_names[yk]
        if yk == 0:
            string = "NN"
        tags.append(string)
        yk = np.int64(yk_sub_1)
    return tags[::-1]


def build_confusion_matrix(test_set, num_tags, all_tags, tag_names, tag_dict):
    confusion_matrix = pd.DataFrame(np.zeros((num_tags, num_tags)), index=tag_names, columns=tag_names)
    for i, line in enumerate(test_set):
        for j, word_tag in enumerate(line):
            word, tag = word_tag
            new_tag = simplify_tag(tag)
            if new_tag in tag_dict:
                confusion_matrix.loc[new_tag, all_tags[i][j]] += 1

    return confusion_matrix


def error_rate_viterbi(all_tags, test_set, word_dict):
    correct_tag_known = 0
    correct_tag_unknown = 0
    counter_known = 0
    counter_unknown = 0
    for i, line in enumerate(test_set):
        for j, word_tag in enumerate(line):
            word, tag = word_tag
            new_tag = simplify_tag(tag)
            if word not in word_dict:
                counter_unknown += 1
                tag_unknown = all_tags[i][j]
                if tag_unknown == new_tag:
                    correct_tag_unknown += 1
            else:
                counter_known += 1
                train_tag = all_tags[i][j]
                if train_tag == new_tag:
                    correct_tag_known += 1

    accuracy_known = correct_tag_known / counter_known
    if counter_unknown == 0:
        accuracy_unknown = 1
    else:
        accuracy_unknown = correct_tag_unknown / counter_unknown
    accuracy_total = (correct_tag_unknown + correct_tag_known) / (counter_known + counter_unknown)
    return accuracy_known, accuracy_unknown, accuracy_total


def calc_unique_words(train_set, test_set):
    words_train_test = defaultdict(int)

    for line in train_set:
        for word, tag in line:
            words_train_test[word] = words_train_test.get(word, 0) + 1

    for line in test_set:
        for word, tag in line:
            words_train_test[word] = words_train_test.get(word, 0) + 1

    return len(words_train_test), sum(words_train_test.values())


if __name__ == '__main__':
    main()
