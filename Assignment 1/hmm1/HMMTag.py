import sys
import re


# Read the emission table from the file
# input: e_file - the emission file
def read_emission_file(e_file):
    e_map = {}
    words = set()
    tags = set()
    word_tag = {}
    with open(e_file, 'r') as emit_file:
        for line in emit_file:
            # split every line and add the
            line_split = line.strip().replace(' ', '').split('=')
            try:
                w, t = line_split[0].split('|')
            except:
                t = line_split[1].split('|')[1]
                line_split = ['=', line_split[2]]
                w = line_split[0]
                if w in word_tag:
                    word_tag[w].append(t)
                else:
                    word_tag[w] = [t]
            words.add(w)
            tags.add(t)
            e_map[(w,t)] = float(line_split[1])
        # add the start and end to the tags list
        tags.add('start')
        tags.add('end')
        words_list = list(words)
        tags_list = list(tags)
        # gives zero value to word|tag not exist in the table
        for w in words_list:
            for t in tags_list:
                if (w,t) not in e_map:
                    e_map[(w,t)] = 0.0
    return e_map, tags_list, word_tag


# Read the transition table from the file
# input: q_file - the transition file
#        extra_file - a file containing extra information
#        tags - a list of existing tags
def read_transition(q_file, extra_file, tags):
    q_map = {}
    # extract the lambda values for extra file
    lambda_vals = open(extra_file, 'r').readline().strip().split()
    lambda_one = float(lambda_vals[0])
    lambda_two = float(lambda_vals[1])
    lambda_three = float(lambda_vals[2])
    first_map = {}
    second_map = {}
    third_map = {}
    # extract the probabilities from every line
    with open(q_file, 'r') as t_file:
        for line in t_file:
            params = re.search('q\((.+?)\|(.+?), (.+?)\) = (.+?) (.+?) (.+?)\n', line)
            first_map[params.group(1) + '_' + params.group(2) + '_' + params.group(3)] = float(params.group(4))
            second_map[params.group(1) + '_' + params.group(3)] = float(params.group(5))
            third_map[params.group(1)] = float(params.group(6))
    # goes through every possibility
    for l1 in tags:
        for l2 in tags:
            for l3 in tags:
                try:
                    second_value = second_map[l1 + '_' + l3]
                except:
                    second_value = 0.0
                try:
                    first_value = first_map[l1 + '_' + l2 + '_' + l3]
                except:
                    first_value = 0.0
                try:
                    third_value = third_map[l1]
                except:
                    third_value = 0.0
                # multiply the values with lambdas values
                q_map[(l2, l3, l1)] = lambda_one * first_value + lambda_two * second_value + lambda_three * third_value
    return q_map


# Gets a list of tags
# input: index - the index of the current iteration
#        tags - a list of the existing tags
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


# return the two words before the current word
# input: sent - the current sentence
#        index - the current loop index
def add_close_words(sent, index):
    if index == 1:
        two_words_before = ['start', 'start']
    elif index == 2:
        two_words_before = ['start', sent[index - 2]]
    else:
        two_words_before = [sent[index - 3], sent[index - 2]]
    return two_words_before


# An implementation of the viterbi algorithm
# input: sent - the current sentence
#        e_map - the emission table
#        q_map - the transition table
#        possible_tags - tags appeared in the training part
#        word_tag - the possible tags for every word map
def viterbi(sent, e_map, q_map, possible_tags, word_tag):
    V = {}
    path = {}
    # initialize the parameters
    V[0, 'start', 'start'] = 1
    path['start', 'start'] = []
    for i in range(1, len(sent) + 1):
        temp_path = {}
        word = sent[i - 1]
        two_words_before = add_close_words(sent, i)
        # check if the current word is known
        if not any(l[0] == word for l in e_map):
            word = change_to_signature(word)
        # gets the possible tags according to the words
        tags_u = get_tags(i - 1, possible_tags, two_words_before[1], word_tag)
        tags_v = get_tags(i, possible_tags, word, word_tag)
        tags_w = get_tags(i - 2, possible_tags, two_words_before[0], word_tag)
        for u in tags_u:
            for v in tags_v:
                V[i, u, v] = 0.0
                prev_w = tags_w[0]
                for w in tags_w:
                    # if the value is 0 there is not need to calculate the value of the viterbi in this point
                    if V[i - 1, w, u] != 0:
                        step = V[i - 1, w, u] * q_map[(w, u, v)] * e_map[(word, v)]
                        if step > V[i, u, v]:
                            V[i, u, v] = step
                            prev_w = w
                temp_path[u, v] = path[prev_w, u] + [v]
        path = temp_path
    tags_u = get_tags(i - 1, possible_tags, two_words_before[1], word_tag)
    tags_v = get_tags(i, possible_tags, word, word_tag)
    prob, umax, vmax = max([(V[len(sent), u, v] * q_map[(u, v, 'end')], u, v) for u in tags_u for v in tags_v])
    return path[umax, vmax]


# Run the viterbi algorithm for every sentence in the file
# input: input_file_name - the test input file
#        e_map - the emission table
#        q_map - the transition table
#        tags - the list of tags exist
def run_viterbi(input_file_name, e_map, q_map, tags, word_tag, output_file):
    tags.remove('end')
    tags.remove('start')
    index = 0
    with open(input_file_name, 'r') as input:
        # get the results of the viterbi algorithm for every sentence and write it to a file
        for sent in input:
            line = sent.strip().split()
            path = viterbi(line, e_map, q_map, tags, word_tag)
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
    input_file_name, q_file, e_file, out_file_name, extra_file_name = argv
    e_map, tags, word_tag = read_emission_file(e_file)
    q_map = read_transition(q_file, extra_file_name, tags)
    run_viterbi(input_file_name, e_map, q_map, tags, word_tag, out_file_name)


if __name__ == '__main__':
    main(sys.argv[1:])