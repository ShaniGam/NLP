import sys


# class containing the maps of words from the training
class Words:

    # constructor
    def __init__(self):
        self.label_word_map = {}
        self.two_words_map = {}
        self.three_words_map = {}

    # Adds the given word to the given map. If the word exists increase the counter by one.
    # input: word - the word we want to add
    #        words_map - the map of words we want to add the word to
    def add_word(self, word, words_map):
        if self.check_if_word_exists(word, words_map):
            words_map[word] += 1.0
        else:
            words_map[word] = 1.0

    # Checks if the word exist in the map
    # input: word - the word we want to check
    #        words_map - a map of words
    def check_if_word_exists(self, word, words_map):
        if word in words_map:
            return True
        else:
            return False

    # Calculates the emission values for every word
    def emission_vals(self):
        emission = {}
        for word in self.label_word_map:
            emission[word] = self.label_word_map[word] * 1.0 / sum(self.label_word_map.itervalues())
        return emission

    # Gets the words map of the tag
    def get_label_word_map(self):
        return self.label_word_map

    # Gets the map of words and counters of A|C
    def get_two_words_map(self):
        return self.two_words_map

    # Gets the map of words and counters of A|B,C
    def get_three_words_map(self):
        return self.three_words_map

    # Gets the value of the given word from the given map
    def get_value_from_map(self, word, words_map):
        return words_map[word]


# Count the # of times the word appears in the file and return the words that appeared once
# input: file name - the name of the text file
def create_words_counter(file_name):
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
    return [k for k,v in words.iteritems() if v < 2]


# change the word to a costumed signature
# input: word - the word we want to change
def change_to_signature(word):
    if len(word) >= 3:
        if word[-2:] == 'ed':
            return 'UNK-ED'
        if word[-3:] == 'ing':
            return 'UNK-ING'
        if word[:2] == 'un':
            return 'UN-UNK'
    return 'UNK'


# assign the words the their tags
# input: word - the current word
#        once_words - the words that appeared once on the text
#        tags_words_map - map of the tags the their words
def words_to_tag(word, once_words, tags_words_map):
    # split to word and its tag
    word_split = word.rsplit('/', 1)
    w = word_split[0]
    # if the word is rare, change it
    if w in once_words:
        w = change_to_signature(w)
    l = word_split[-1]
    if l not in tags_words_map:
        tags_words_map[l] = Words()
    tags_words_map[l].add_word(w, tags_words_map[l].get_label_word_map())


# number of words in the training file
words_counter = 0


# Reads the words from the file and map the words and their tags
# input: file name - the name of the text file
def read_data(file_name):
    tags_list = {}
    tags_words_map = {}
    # words that appeared once of the training file
    once_words = create_words_counter(file_name)
    with open(file_name, "r") as train_file:
        for line in train_file:
            split_line = line.strip().split()
            global words_counter
            words_counter += len(split_line)
            # for every word in the current sentence
            for word in split_line:
                words_to_tag(word, once_words, tags_words_map)
            tags = [i.rsplit('/', 1)[1] for i in split_line]
            # insert start and end to the tags list
            for i in range(2):
                tags.insert(0, 'start')
            tags.append('end')
            first = tags[0]
            if first not in tags_list:
                tags_list[tags[0]] = Words()
            tags_list[first].add_word(first, tags_list[first].get_two_words_map())
            # save the tag before and the two tags before every third tag
            for first, second, third in (tags[i:i + 3] for i in range(len(tags) - 2)):
                if third not in tags_list:
                    tags_list[third] = Words()
                tags_list[third].add_word(second, tags_list[third].get_two_words_map())
                tags_list[third].add_word(first + '_' + second, tags_list[third].get_three_words_map())
    train_file.close()
    return tags_words_map, tags_list


# Calculate the transition table
# input: tags_list - list of all the tags exist in the training file and their words
def calculate_tran(tags_list):
    tran_map = {}
    for tag in tags_list:
        first_list = {}
        second_list = {}
        current_word = tags_list[tag]
        # gets the map of all the tags appeared before the current tag
        two_words_map = current_word.get_two_words_map()
        three_words_map = current_word.get_three_words_map()
        for w in three_words_map:
            tag_split = w.split('_')
            first = tag_split[0]
            second = tag_split[1]
            # gets the number of times first and second appeared before the current word
            calc_first = current_word.get_value_from_map(w, three_words_map)
            calc_second = current_word.get_value_from_map(second, two_words_map)
            if len(tags_list[second].get_two_words_map()) != 0:
                # calculate the probabilities
                calc_first /= tags_list[second].get_value_from_map(first, tags_list[second].get_two_words_map())
                calc_second /= sum(tags_list[second].get_two_words_map().itervalues())
            first_list[w] = calc_first
            second_list[second] = calc_second
        # the number of times the tag appeared / the total number of words
        third_val = sum(two_words_map.itervalues()) / words_counter
        tran_map[tag] = [first_list, second_list, third_val]
    return tran_map


# Calculate the emission table
# input: train - list of labels in the training file
def calculate_emission(train):
    emit = {}
    for label in train:
        emit[label] = train[label].emission_vals()
    return emit


# Write the emission table to a file
# input: emit_table - the emission table
#        e_file - the name of the emission file
def emission_to_file(emit_table, e_file):
    emit_file = open(e_file, 'w+')
    for label in emit_table:
        words = emit_table[label]
        for word in words:
            emit_file.write(word + "|" + label + " = " + str(words[word]))
            emit_file.write('\n')
    emit_file.close()


# Write the transition table to a file
# input: trans_table - the transition table
#        q_file - the name of the transition file
def transition_to_file(trans_table, q_file):
    trans_file = open(q_file, 'w+')
    for word in trans_table:
        first_result = trans_table[word][0]
        second_result = trans_table[word][1]
        for item in first_result:
            split_word = item.split('_')
            trans_file.write("q(" + word + '|' + item.replace('_', ', ') + ') = ' + str(first_result[item])
                             + ' ' + str(second_result[split_word[1]]) + ' ' + str(trans_table[word][2]))
            trans_file.write('\n')
    trans_file.close()


def main(argv):
    input_file_name, q_file, e_file = argv
    train, tags_list = read_data(input_file_name)
    emit_table = calculate_emission(train)
    emission_to_file(emit_table, e_file)
    tran_table = calculate_tran(tags_list)
    transition_to_file(tran_table, q_file)


if __name__ == '__main__':
    main(sys.argv[1:])
