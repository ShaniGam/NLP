import sys
import json
import liblin as ll


# Read the tags, features and word_tag maps from the extra file
# input: extra_file - the name of the extra information file
def read_extra_file(extra_file):
    with open(extra_file, 'r') as f:
        json_data = [json.loads(line) for line in f]
        tags = json_data[0][0]
        features = json_data[0][1]
        word_tag = json_data[0][2]
    f.close()
    # change the maps to string and int format
    tags_str = {str(k): str(v) for v, k in tags.items()}
    features_str = {str(k): int(v) for k, v in features.items()}
    word_tag_str = {str(k): [str(v) for  v in word_tag[k]] for k in word_tag.keys()}
    return tags_str, features_str, word_tag_str


# Add the feature to the word' features vector
# input: feature - the feature we want to add to the vector
#        features_map - the features we found in the training precess
#        vec - the features vector
def add_to_vec(feature, features_map, vec):
    if feature in features_map:
        vec.append(features_map[feature])


# Remove the feature from the features vector of the word
# input: feature - the feature we want to remove from the vector
#        features_map - the features we found in the training precess
#        vec - the features vector
def remove_from_vec(feature, features_map, vec):
    if feature in features_map:
        if features_map[feature] in vec:
            vec.remove(features_map[feature])


# Create a feature vector for the given word
# input: two_words_before - the two words before the current word
#        two_words_after - the two words after the current word
#        features - the features we found on the training process
def get_features_vectors(two_words_before, word, two_words_after, features):
    word_vec = []
    # add the words around the current word as features
    add_to_vec('wi-1_' + two_words_before[1], features, word_vec)
    add_to_vec('wi-2_' + two_words_before[0], features, word_vec)
    add_to_vec('wi+1_' + two_words_after[0], features, word_vec)
    add_to_vec('wi+2_' + two_words_after[1], features, word_vec)
    add_to_vec('wi_' + word, features, word_vec)

    # add the suffix and prefix of every word to the features vec
    prefix = suffix = ''
    word_len = len(word)
    loop_range = min(word_len, 4)
    for i in range(loop_range):
        prefix += word[i]
        add_to_vec('p_' + prefix, features, word_vec)
        suffix = word[word_len - 1 - i] + suffix
        add_to_vec('s_' + suffix, features, word_vec)

    # check if the word contains certain characters
    contains_num = any(char.isdigit() for char in word)
    if contains_num:
        add_to_vec('contains_number', features, word_vec)
    contains_uppercase = any(char.isupper() for char in word)
    if contains_uppercase:
        add_to_vec('contains_upper_case', features, word_vec)
    hyphen = '-'
    if hyphen in word:
        add_to_vec('contains_hyphen', features, word_vec)
    return word_vec


# Gets a list of tags
# input: index - the index of the current iteration
#        tags - a list of the existing tags
#        word - the current word
#        word_tag - the map of the words and their possible tags
def get_tags(index, tags, word, word_tag):
    if index == -1:
        return ['start']
    if index == 0:
        return ['start']
    else:
        # pruning - return only the tags relevant to the specific word
        if word in word_tag:
            return word_tag[word]
        else:
            return tags


# return the two words before the current word
# input: sent - the current sentence
#        sent_len - the number of words in the sentence
#        index - the current loop index
def add_close_words(sent, sent_len, index):
    # gets the two words before the current word
    if index == 1:
        two_words_before = ['start', 'start']
    elif index == 2:
        two_words_before = ['start', sent[index - 2]]
    else:
        two_words_before = [sent[index - 3], sent[index - 2]]
    # gets the two words after the current word
    two_words_after = []
    if index == sent_len:
        two_words_after = ['end', 'end']
    elif index == sent_len - 1:
        two_words_after = [sent[sent_len - 1], 'end']
    elif index <= sent_len - 2:
        two_words_after = [sent[index], sent[index + 1]]
    return two_words_before, two_words_after


# Calculate the score of the current viterbi step
# input: w, u, v - the current tag and the two tags before
#        features_map - the features we found in the training precess
#        word_vec - the features vector
#        llp - the log linear predictor
#        tags_map - the tags-number mapping
#        step_before - the result of the step before's score
def calc_score(w, u, v, features_map, word_vec, llp, tags_map, step_before):
    two_before = 'ti-2ti-1_' + w + '_' + u
    add_to_vec(two_before, features_map, word_vec)
    score = step_before * ll.prediction(word_vec, llp)[tags_map[v]]
    remove_from_vec(two_before, features_map, word_vec)
    return score


# An implementation of the viterbi algorithm
# input: sent - the current sentence
#        features_map - the features we found in the training precess
#        tags_map - the tags-number mapping
#        llp - the log linear predictor
#        word_tag - the map of the words and their possible tags
def viterbi(sent, features_map, tags_map, llp, word_tag):
    V = {}
    path = {}
    # initialize the parameters
    V[0, 'start', 'start'] = 1
    path['start', 'start'] = []
    # possible_tags = [str(i) for i in tags.values()]
    possible_tags = tags_map.keys()
    sent_len = len(sent)
    for i in range(1, sent_len + 1):
        temp_path = {}
        two_words_before, two_words_after = add_close_words(sent, sent_len, i)
        word = sent[i - 1]
        # gets the possible tags according to the words
        word_vec = get_features_vectors(two_words_before, word, two_words_after, features_map)
        tags_u = get_tags(i - 1, possible_tags, two_words_before[1], word_tag)
        tags_v = get_tags(i, possible_tags, word, word_tag)
        tags_w = get_tags(i - 2, possible_tags, two_words_before[0], word_tag)
        # the tag of the word before
        for u in tags_u:
            add_to_vec('ti-1_' + u, features_map, word_vec)
            # the tag of the current word
            for v in tags_v:
                V[i, u, v] = 0.0
                prev_w = tags_w[0]
                # the tag of the two words before
                for w in tags_w:
                    step = calc_score(w, u, v, features_map, word_vec, llp, tags_map, V[i - 1, w, u])
                    # get the maximum value
                    if step > V[i, u, v]:
                        V[i, u, v] = step
                        prev_w = w
                temp_path[u, v] = path[prev_w, u] + [v]
            remove_from_vec('ti-1_' + u, features_map, word_vec)
        path = temp_path
    # the final step
    max_val = 0.0
    tags_u = get_tags(i - 1, possible_tags, two_words_before[1], word_tag)
    tags_v = get_tags(i, possible_tags, word, word_tag)
    for u in tags_u:
        add_to_vec('ti-1_' + u, features_map, word_vec)
        for v in tags_v:
            step = calc_score(u, v, v, features_map, word_vec, llp, tags_map, V[len(sent), u, v])
            if step > max_val:
                max_val = step
                umax = u
                vmax = v
        remove_from_vec('ti-1_' + u, features_map, word_vec)
    return path[umax, vmax]


# Run the viterbi algorithm for every sentence in the file
# input: input_file_name - the test input file
#        tags - the list of tags exist
#        features_map - the features we found in the training precess
#        word_tag - the map of the words and their possible tags
def run_viterbi(input_file_name, tags, features, word_tag, output_file, model):
    llp = ll.LiblinearLogregPredictor(model)
    index = 0
    with open(input_file_name, 'r') as input:
        for sent in input:
            line = sent.strip().split()
            path = viterbi(line, features, tags, llp, word_tag)
            index += 1
            write_to_file(output_file, line, path)
    input.close()


# Write the results to a file
# input: output_file - the file we want to write the results to
#        line - the words of the current sentence
#        path - the results of the viterbi algorithm
def write_to_file(output_file, line, path):
    with open(output_file, 'a+') as output:
        str_sent = ''
        for i in range(len(path) - 1):
            str_sent += line[i] + '/' + path[i] + ' '
        str_sent += line[len(path) - 1] + '/' + path[len(path) - 1]
        output.write(str_sent)
        output.write('\n')
    output.close()


def main(argv):
    input_file_name, model_file_name, output_file_name, extra_file_name = argv
    tags, features, word_tag = read_extra_file(extra_file_name)
    run_viterbi(input_file_name, tags, features, word_tag, output_file_name, model_file_name)


if __name__ == '__main__':
    main(sys.argv[1:])