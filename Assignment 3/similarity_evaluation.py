import numpy as np
from collections import defaultdict, Counter,OrderedDict

NOUN_TAG_SET = set(['NN', 'NNS', 'NNP', 'NNPS'])
CONTENT_TAG_SET = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG',
                       'VBN', 'VBP', 'VBZ', 'WRB'])
PREPOSITION_TAG = 'IN'
global common_target_threshold
global common_attribute_threshold
common_target_threshold = 100
common_attribute_threshold = 20

# Convert a vector of number to dictionary
# vec - vector of numbers
def vec_to_dict(vec):
    prop = {}
    prop['ID'] = vec[0]
    prop['LEMMA'] = vec[2]
    prop['CPOSTAG'] = vec[3]
    prop['HEAD'] = vec[6]
    prop['DEPREL'] = vec[7]
    return prop


# Each element in data represents a sentence and each element in sentence represents a word with its fields
# data_fname = the file name of the corpus file
def load_data(data_fname):
    data = []
    row = []
    W2I = {}
    index = 0
    words_count = {}
    for line in file(data_fname):
        text = line.strip()
        if not text:
            data.append(row)
            row =[]
        else:
            word = vec_to_dict(text.split('\t'))
            # map the lemma values of the words
            if word['LEMMA'] not in W2I:
                W2I[word['LEMMA']] = index
                index += 1
            current_word_index = W2I[word['LEMMA']]
            if current_word_index in words_count:
                words_count[current_word_index] += 1
            else:
                words_count[current_word_index] = 1
            row.append(word)
    return data, W2I ,words_count


# Add feature to a word using the window method
# sent - a sentence from the corpus
# word_count - a map of words and counters
# W2I - a map with every word and its index
# i - the current word position
def add_feature_window(sent, word_count, W2I, i):
    word = sent[i]['LEMMA']
    if W2I[word] not in word_count:
        word_count[W2I[word]] = 1
    else:
        word_count[W2I[word]] += 1
    global words_counter
    words_counter += 1


# Add a word to the relevant feature in the map
# context - the relevant feature
# att_words_mapping - attributes-words map
# current_word - the current word we want to add
def add_context(context,att_words_mapping,current_word):
    if context in att_words_mapping:
        att_words_mapping[context].add(current_word)
    else:
        att_words_mapping[context] = set([current_word])


# Create a map with content words within a window of two content words on each side of the target word
# data - the words on the corpus
# W2I - a map with every word and its index
# words_count - a map of words and counters
def window_vec(data, W2I, words_count):
    words = {}
    att_words_mapping = {}
    for sent in data:
        sent_len = len(sent)
        for i in range(sent_len):
            current_word = W2I[sent[i]['LEMMA']]
            # add the feature of the word only it appeared at least 100 times
            if words_count[current_word] >= common_target_threshold:
                # extract the map of features of the word if exists
                if current_word in words:
                    word_count = words[current_word]
                else:
                    word_count = words[current_word] = {}
                # add every words in the window as feature if it appears at least 20 times
                if i > 0:
                    context = W2I[sent[i - 2]['LEMMA']]
                    if words_count[context] >= common_attribute_threshold:
                        add_feature_window(sent, word_count, W2I, i - 2)
                        add_context(context, att_words_mapping, current_word)
                if i > 1:
                    context = W2I[sent[i - 1]['LEMMA']]
                    if words_count[context] >= common_attribute_threshold:
                        add_feature_window(sent, word_count, W2I, i - 1)
                        add_context(context, att_words_mapping, current_word)
                if i < sent_len - 1:
                    context = W2I[sent[i + 1]['LEMMA']]
                    if words_count[context] >= common_attribute_threshold:
                        add_feature_window(sent, word_count, W2I, i + 1)
                        add_context(context, att_words_mapping, current_word)
                if i < sent_len - 2:
                    context = W2I[sent[i + 2]['LEMMA']]
                    if words_count[context] >= common_attribute_threshold:
                        add_feature_window(sent, word_count, W2I, i + 2)
                        add_context(context, att_words_mapping, current_word)

    return words, att_words_mapping


# the current feature index
features_index = 0


# Add feature to a word using the dependency edges method
# kind - child/parent
# word_count - a map of words and counters
# item1 - the first item we want to add to the feature
# item2 - the second item we want to add to the feature
# features_map - a map with all the features and their indexes
# att_words_mapping - attributes-words map
# current_word - the current words we want to add the feature to
def add_feature_tree(kind, item1, item2, features_map, word_count, att_words_mapping, current_word):
    global features_index
    feature = kind + item1 + '_' + item2
    # add the feature to the feature map
    if feature not in features_map:
        features_map[feature] = features_index
        features_index += 1
    f = features_map[feature]
    # count the number of the times the feature appeared with the word
    if f not in word_count:
        word_count[f] = 1
    else:
        word_count[f] += 1
    if f not in att_words_mapping:
        att_words_mapping[f] = set()
    att_words_mapping[f].add(current_word)
    global words_counter
    words_counter += 1


# Create a map of words and features using the dependency edges method
# data - the words on the corpus
# W2I - a map with every word and its index
# words_count - a map of words and counters
def connected_words_vec(data, W2I, words_count):
    words = {}
    features_map = {}
    att_words_mapping = {}
    for sent in data:
        sent_len = len(sent)
        for i in range(sent_len):
            current_word = W2I[sent[i]['LEMMA']]
            word_tag = sent[i]['CPOSTAG']
            if words_count[current_word] >= common_target_threshold and word_tag in CONTENT_TAG_SET:
                if current_word in words:
                    word_count = words[current_word]
                else:
                    word_count = words[current_word] = {}

                # check the parent of the current node
                head = int(sent[i]['HEAD'])
                if head != 0:
                    parent_word = sent[head - 1]
                    # if the parent is a preposition
                    if parent_word['CPOSTAG'] == 'IN':
                        ancestor = sent[int(parent_word['HEAD']) - 1]
                        if words_count[W2I[ancestor['LEMMA']]] >= common_attribute_threshold and ancestor['CPOSTAG'] in CONTENT_TAG_SET:
                            prep = parent_word['DEPREL'] + '-' + parent_word['LEMMA']
                            add_feature_tree('child_', prep, ancestor['LEMMA'], features_map, word_count, att_words_mapping, current_word)
                    elif words_count[W2I[parent_word['LEMMA']]] >= common_attribute_threshold and parent_word['CPOSTAG'] in CONTENT_TAG_SET:
                        add_feature_tree('child_', sent[i]['DEPREL'], parent_word['LEMMA'], features_map, word_count,
                                         att_words_mapping, current_word)
                # check the children of the current node
                for j in range(sent_len):
                    parent_id = int(sent[j]['HEAD'])
                    word_id = int(sent[i]['ID'])
                    if parent_id == word_id:
                        child = sent[j]['DEPREL']
                        # if the child is a preposition
                        if sent[j]['CPOSTAG'] == 'IN':
                            for k in range(sent_len):
                                if int(sent[k]['HEAD']) == int(sent[j]['ID']) and sent[k]['CPOSTAG'] in NOUN_TAG_SET:
                                    prep_child = sent[k]
                                    if words_count[W2I[prep_child['LEMMA']]] >= common_attribute_threshold and prep_child['CPOSTAG'] in CONTENT_TAG_SET:
                                        prep = sent[j]['DEPREL'] + '-' + sent[j]['LEMMA']
                                        add_feature_tree('parent_', prep, prep_child['LEMMA'], features_map, word_count, att_words_mapping, current_word)
                                    break
                        elif words_count[W2I[sent[j]['LEMMA']]] >= common_attribute_threshold and sent[j]['CPOSTAG'] in CONTENT_TAG_SET:
                            add_feature_tree('parent_', child, sent[j]['LEMMA'], features_map, word_count, att_words_mapping,
                                             current_word)
    return words, att_words_mapping

# Create a map with content words within a sentence
# data - the words on the corpus
# W2I - a map with every word and its index
# words_count - a map of words and counters
def extract_vectors_by_sentence(data,W2I,words_count):
    counts = defaultdict(Counter)
    #with which words the att appears
    att_words_mapping = {}
    all_pairs = 0
    for sen in data:
        for i in range(0,len(sen)):
            current_word = W2I[sen[i]['LEMMA']]
            if words_count[current_word] >= common_target_threshold:
                for j in range(0,len(sen)):
                    if j != i:
                        context = W2I[sen[j]['LEMMA']]
                        if words_count[context] >= common_attribute_threshold:
                            context_counts_for_word = counts[current_word]
                            context_counts_for_word[context] += 1
                            all_pairs += 1
                            #add to attribute mapping only words that appear with  this context as attribute
                            if context in att_words_mapping:
                                att_words_mapping[context].add(current_word)
                            else:
                                att_words_mapping[context] = set([current_word])
    return counts,all_pairs,att_words_mapping


# Compute the pmi value for every word
def PMI(words, attributes_mapping):
    one_word = {}
    two_words = {}
    one_attribute = {}
    # probability of every word
    for word in words:
        word_map = words[word]
        one_word[word] = sum(word_map.itervalues()) * 1.0 / words_counter
        for att in word_map:
             two_words[(word, att)] = (word_map[att] * 1.0 / words_counter)

    # probability of every attribute
    for att in attributes_mapping:
        counts = []
        words_list = attributes_mapping[att]
        for word in words_list:
            word_map = words[word]
            count = word_map[att]
            counts.append(count)
        count_sum = sum(counts)
        one_attribute[att] = count_sum * 1.0 / words_counter
    words_pmi = {}
    # pmi values
    for word in words:
        pmi_word = words_pmi[word] = {}
        word_map = words[word]
        for att in word_map:
            pmi_word[att] = np.log(two_words[(word, att)] / (one_word[word] * one_attribute[att]))
    return words_pmi

# Compute the similarity vector for u (the target word)
# words_pmi - the pmi values for each word and attribute
# attributes_mapping - map each attribute to the words that it appear with
def compute_similarty(u,words_pmi,attributes_mapping):
    u_att_list = words_pmi[u]
    DT = {}
    for att in u_att_list:
        possible_words = find_possible_words(att,attributes_mapping)
        for v in possible_words:
            key = (u,v)
            v_att_list = words_pmi[v]
            if key in DT:
                DT[key] += u_att_list[att] * v_att_list[att]
            else:
                DT[key] = u_att_list[att] * v_att_list[att]
    return DT


# find all the words with the attribute att and appear at least 100 times in corpus
def find_possible_words(att,attributes_mapping):
    att_words_set = attributes_mapping[att]
    return att_words_set

# normalize the results in the similarity vector
def normailze_results(results,words_pmi):
    for key in results.keys():
        u,v = key
        u_sum =0.0
        v_sum = 0.0
        for pmi in words_pmi[u]:
            u_sum+=pmi**2
        for pmi in words_pmi[v]:
            v_sum+=pmi**2
        u_sum = np.sqrt(u_sum)
        v_sum = np.sqrt(v_sum)
        results[key] /= (u_sum * v_sum)


def check_window_vec(data, W2I, words_count):
    global words_counter
    words_counter = 0
    words, attributes_mapping = window_vec(data, W2I, words_count)
    find_similar_words(words, attributes_mapping, 'evaluation_window')


def check_sentence_vec(data, W2I, words_count):
    words, all_pairs, attributes_mapping = extract_vectors_by_sentence(data, W2I, words_count)
    global words_counter
    words_counter = all_pairs
    find_similar_words(words, attributes_mapping, 'evaluation_sentence')


def check_tree_vec(data, W2I, words_count):
    global words_counter
    words_counter = 0
    words, attributes_mapping = connected_words_vec(data, W2I, words_count)
    find_similar_words(words, attributes_mapping, 'evaluation_tree')

# find 20 similar words for each word in the words list and write the results into a file
def find_similar_words(words, attributes_mapping, file_name):
    words_pmi = PMI(words, attributes_mapping)
    out = open(file_name, 'w+')
    for target_word in word_list:
        words_str = ' '
        sim_words = []
        results = compute_similarty(W2I[target_word], words_pmi, attributes_mapping)
        normailze_results(results, words_pmi)
        ordered_results = sorted(results, key=results.get, reverse=True)[:21]
        out.write('similar words of word ' + target_word + '\n')
        for key in ordered_results:
            if W2I[target_word] != key[1]:
                sim_words.append(IW2[key[1]])
        words_str = words_str.join(sim_words)
        out.write(words_str + '\n')
    out.close


if __name__ == '__main__':
    import sys
    data, W2I, words_count = load_data(sys.argv[1])
    IW2 = dict((v, k) for k, v in W2I.iteritems())
    word_list = ['car','bus','hospital','hotel','gun','bomb','horse','fox','table','bowl','guitar','piano']
    check_sentence_vec(data, W2I, words_count)
    check_window_vec(data, W2I, words_count)
    check_tree_vec(data, W2I, words_count)