import sys
import json


# Count the # of times the word appears in the file and return the words that appeared once
# input: file name - the name of the text file
def create_rare_words_counter(file_name):
    words = {}
    with open(file_name, "r") as train_file:
        for line in train_file:
            split_line = line.strip().split()
            # for every word in the text
            for word in split_line:
                word_split = word.rsplit('/', 1)
                w = word_split[0]
                # if the word already appeared, increase its counter by 1
                if w in words:
                    words[w] += 1
                # else add the word to the dictionary with value 1
                else:
                    words[w] = 1
    train_file.close()
    # return the word that appeared less than twice in the text
    return [k for k,v in words.iteritems() if v < 5]

# the current feature index
features_index = 1


# Add the feature to the features vector of the word and to the features dictionary
def add_feature(item, kind, features, feature_vec):
    f = kind + '_' + item
    global features_index
    # if the word is already exists in the dictionary
    if f in features:
        feature_vec.append(features[f])
    else:
        # add the word to the vector and the dictionary
        features[f] = features_index
        feature_vec.append(features_index)
        features_index += 1


# Add the word and tag to the dictionary
# input: word - the current word
#        tag - the tag of the current word
#        word_tag - a dictionary
def add_word_tag(word, tag, word_tag):
    if word in word_tag:
        if tag not in word_tag[word]:
            word_tag[word].append(tag)
    else:
        word_tag[word] = [tag]


# Add the basic features of the word to its features vector
# input: first - the word which is two words before the current word
#        second - the word before the current word
#        fourth - the word after the current word
#        fifth - the word which is two words after the current word
#        features - the features dictionary
#        feature_vec - the features vector of the current word
def add_basic_features(first, second, fourth, fifth, features, feature_vec):
    add_feature(second.rsplit('/', 1)[0], 'wi-1', features, feature_vec)
    add_feature(first.rsplit('/', 1)[0], 'wi-2', features, feature_vec)
    add_feature(fourth.rsplit('/', 1)[0], 'wi+1', features, feature_vec)
    add_feature(fifth.rsplit('/', 1)[0], 'wi+2', features, feature_vec)
    add_feature(second.rsplit('/', 1)[1], 'ti-1', features, feature_vec)
    add_feature(first.rsplit('/', 1)[1] + '_' + second.rsplit('/', 1)[1], 'ti-2ti-1', features, feature_vec)


# Add extra features of the word to its features vector
# input: word - the current word
#        rare_words - a list of words that appeared less than 5 times on the text
#        features - the features dictionary
#        feature_vec - the features vector of the current word
def add_extra_features(word, rare_words, features, feature_vec):
    if word not in rare_words:
        # add the current word
        add_feature(word, 'wi', features, feature_vec)
    else:
        # add the suffix and prefix of every word to the features vec
        prefix = suffix = ''
        word_len = len(word)
        loop_range = min(word_len, 4)
        for i in range(loop_range):
            prefix += word[i]
            add_feature(prefix, 'p', features, feature_vec)
            suffix = word[word_len - 1 - i] + suffix
            add_feature(suffix, 's', features, feature_vec)

        # check if the word contains certain characters
        contains_num = any(char.isdigit() for char in word)
        if contains_num:
            add_feature('number', 'contains', features, feature_vec)
        contains_uppercase = any(char.isupper() for char in word)
        if contains_uppercase:
            add_feature('upper_case', 'contains', features, feature_vec)
        hyphen = '-'
        if hyphen in word:
            add_feature('hyphen', 'contains', features, feature_vec)


# Create a features vector for every word
# input: file_name - the name of the train file
def create_features_vec(file_name):
    rare_words = create_rare_words_counter(file_name)
    tags_map = {}
    features = {}
    word_tag = {}
    with open(file_name, "r") as train_file:
        for line in train_file:
            split_line = line.strip().split()
            # insert start and end to the tags list
            for i in range(2):
                split_line.insert(0, 'start/start')
                split_line.append('end/end')
            for first, second, third, fourth, fifth in (split_line[i:i + 5] for i in range(len(split_line) - 4)):
                feature_vec = []
                add_basic_features(first, second, fourth, fifth, features, feature_vec)
                word = third.rsplit('/', 1)[0]
                tag = third.rsplit('/', 1)[1]
                add_word_tag(word, tag, word_tag)
                add_extra_features(word, rare_words, features, feature_vec)
                # add the features vector to the map according to its tag
                if tag in tags_map:
                    tags_map[tag].append(feature_vec)
                else:
                    tags_map[tag] = [feature_vec]
    return features, tags_map, word_tag


# Write the tag and features vector of every word to a file
# input: output_file_name - the name of the output file
#        tags_map - a map with all the features vectors of every tag
def write_to_file(output_file_name, tags_map):
    tags = {}
    output_file = open(output_file_name, "w+")
    tag_index = 1
    for tag in tags_map:
        tags[tag_index] = tag
        # the features vectors of the words with the current tag
        words = tags_map[tag]
        for word_features in words:
            # create a line containing the tag and the features of the word
            word_features.sort()
            tag_str = str(tag_index)
            for w in word_features:
                tag_str += ' ' + str(w) + ':' + '1'
            output_file.write(tag_str)
            output_file.write('\n')
        tag_index += 1
    output_file.close()
    return tags


# Write the feature-number mapping to a json file
# input: features - the features-numbers mapping
#        tags - the tags-numbers mapping
#        output_file - the name of the file we want to write to
def write_features_map(features, tags, output_file):
    data = []
    data.append(tags)
    data.append(features)
    with open(output_file, 'w+') as f:
        json.dump(data, f)
    f.close()


# Write the feature-number mapping and the word-tag map to a json file
# input: features - the features-numbers mapping
#        tags - the tags-numbers mapping
#        word_tag - a map with every word and its tags
#        output_file - the name of the file we want to write to
def write_extra_file(word_tag, features, tags, output_file):
    data = []
    data.append(tags)
    data.append(features)
    data.append(word_tag)
    with open(output_file, 'w+') as f:
        json.dump(data, f)
    f.close()


def main(argv):
    input_file_name, memm_train, output_features_file, extra_file = argv
    features, tags_map, word_tag = create_features_vec(input_file_name)
    tags = write_to_file(memm_train, tags_map)
    write_features_map(features, tags, output_features_file)
    write_extra_file(word_tag, features, tags, extra_file)


if __name__ == '__main__':
    main(sys.argv[1:])
