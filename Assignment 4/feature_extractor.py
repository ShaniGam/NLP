import codecs
import spacy

from utils import load_annotated_file, write_annotated_file,write_features_file, load_features_file
from classes import Candidate, Entity
from csr_converter import create_sparse_feature_vectors
from classifier import train,predict
from sklearn.metrics import classification_report

nlp = spacy.load('en')


def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        yield sent_id, sent


features_index = 0
features_map = {}


'''
This function add feature to candidate's features vector and to the features map (if needed)
'''
def add_feature(candidate, f, in_train):
    global features_index
    # if the word is already exists in the dictionary
    if f in features_map:
        candidate.add_feature(features_map[f])
    elif in_train:
        # add the word to the vector and the dictionary
        features_map[f] = features_index
        candidate.add_feature(features_map[f])
        features_index += 1


'''
This function create a features of dependency path (dependency labels and path distance ) between two entities
'''
def find_dep_path(candidate, sent, in_train):
    stopped_by_root = True
    entity1 = candidate.entity1.entity
    entity2 = candidate.entity2.entity
    if entity1.root.i < entity2.root.i:
        ent1_first = True
    else:
        ent1_first = False
    i = entity1.root.i
    ent1_to_root = []
    while sent[i].dep_ != 'ROOT':  # try to find path to root
        ent1_to_root.append(sent[i].dep_)
        i = sent[i].head.i
        if i == entity2.root.i:  # stop if find path to entity 2 without root
            stopped_by_root = False
            ent1_to_root.append(sent[i].dep_)
            break

    if stopped_by_root:
        ent2_to_root = []
        i = entity2.root.i
        while sent[i].dep_ != 'ROOT':  # try to find path to root
            ent2_to_root.append(sent[i].dep_)
            i = sent[i].head.i
            if i == entity1.root.i:  # stop if find path to entity 1 without root
                stopped_by_root = False
                ent2_to_root.append(sent[i].dep_)
                break
        if stopped_by_root:
            if ent1_first:  # make the dependency labels be in the same order
                ent1_to_root.append('ROOT')
                path = ent1_to_root + ent2_to_root[::-1]
            else:
                ent2_to_root.append('ROOT')
                path = ent2_to_root + ent1_to_root[::-1]
        else:
            path = ent2_to_root
    else:
        path = ent1_to_root

    if not path:
        print 'empty path'

    path_string = 'dep_path_' + '_'.join(path)
    dependency_dist = 'dep_distance_' + str(len(path))
    add_feature(candidate, dependency_dist, in_train)
    add_feature(candidate, path_string, in_train)


'''
This function create a features of distance between two entities
'''
def create_distances_features(candidate, in_train):
    if candidate.entity2.entity.start > candidate.entity1.entity.start:
        entities_distance = 'distance_' + str(candidate.entity2.entity.start - candidate.entity1.entity.end)
        distance = candidate.entity2.entity.start - candidate.entity1.entity.end
    else:
        entities_distance = 'distance_' + str(candidate.entity1.entity.start - candidate.entity2.entity.end)
        distance = candidate.entity1.entity.start - candidate.entity2.entity.end
    add_feature(candidate, entities_distance, in_train)
    if distance == 1:
        add_feature(candidate, 'one_word_distance', in_train)


'''
 This function create a features of entities's information combinations
'''
def entities_combinations(candidate, i, j, in_train):
    distance = abs(i - j) - 1
    distance_feature = 'ent-dist_' + str(distance)
    add_feature(candidate, distance_feature, in_train)
    label = candidate.entity1.entity.label_ + '_' + candidate.entity2.entity.label_
    add_feature(candidate, label, in_train)
    pos = candidate.entity1.entity.root.pos_ + '_' + candidate.entity2.entity.root.pos_
    add_feature(candidate, pos, in_train)
    dep = candidate.entity1.entity.root.dep_ + '_' + candidate.entity2.entity.root.dep_
    add_feature(candidate, dep, in_train)


'''
This function check if two entities is in the same NP chunk (and add feature to candidate if needed)
'''
def same_np_chunk(candidate, sent, in_train):
    for np in sent.noun_chunks:
        entity_text1 = candidate.entity1.entity.text
        entity_text2 = candidate.entity2.entity.text
        if entity_text1 in np.text and entity_text2 in np.text and np.text.index(entity_text1) != np.text.index(
                entity_text2):
            add_feature(candidate, 'same_np_chunk', in_train)
            break


'''
This function create the environmental features of a candidate (two entities in the same sentence)
'''
def create_environmental_features(sent, candidate, in_train, candidate_distance):
    has_comma(sent, candidate, in_train, candidate_distance)
    same_np_chunk(candidate, sent, in_train)
    create_distances_features(candidate, in_train)
    find_dep_path(candidate, sent, in_train)

    if candidate.entity2.entity.start > candidate.entity1.entity.start:
        add_feature(candidate, 'ent1_first', in_train)


'''
This function add a features of apposition and if there a comma between two entities
    and check the apposition rule that we added
'''
def has_comma(sent, candidate, in_train, candidate_distance):
    comma_counter = 0
    if candidate.entity2.entity.start > candidate.entity1.entity.start:
        start = candidate.entity1.entity.end
        end = candidate.entity2.entity.start
        ent1_first = True
    else:
        start = candidate.entity2.entity.end
        end = candidate.entity1.entity.start
        ent1_first = False

    for i in range(start,end):
        if sent[i].text == ',':
            comma_counter += 1

    if comma_counter >= 1:
        add_feature(candidate, 'comma_between_entities', in_train)
        if ent1_first:
            if candidate.entity1.entity.label_ == 'PERSON' and candidate.entity2.entity.label_ == 'ORG':
                dist = candidate.entity2.entity.start - candidate.entity1.entity.end
                start = candidate.entity1.entity.end + 1
                end = candidate.entity2.entity.start
                appos = False
                for index in range(start,end):
                    if sent[index].dep_ == 'appos':
                        appos = True
                if sent[candidate.entity2.entity.end].dep_ == 'appos':
                    appos = True
                if appos:
                    add_feature(candidate, 'apposition', in_train)
                exist = False
                if sent[candidate.entity1.entity.end].text == ',' and (appos or sent[start].text == 'a' or sent[start].text == 'an')and dist < 10:
                    for k in candidate_distance:
                        if k.entity1.text == candidate.entity1.text:
                            if candidate_distance[k] > dist:
                                del(candidate_distance[k])
                                candidate_distance[candidate] = dist
                            exist = True
                    if not exist:
                        candidate_distance[candidate] = dist


'''
This function create a features in the entity level (information about specific entity)
'''
def create_entities_features(sent, candidate, entity, index, in_train):
    if index == 0:
        pos = 'left'
    else:
        pos = 'right'
    if entity.start != 0:
        word_before_feat = pos + '_wi-1_' + sent[entity.start - 1].lemma_
        add_feature(candidate, word_before_feat, in_train)
        tag_before_feat = pos + '_ti-1_' + sent[entity.start - 1].tag_
        add_feature(candidate, tag_before_feat, in_train)
    if entity.end < len(sent):
        word_after_feat = pos + '_wi+1_' + sent[entity.end].lemma_
        add_feature(candidate, word_after_feat, in_train)
        tag_after_feat = pos + '_ti+1_' + sent[entity.end].tag_
        add_feature(candidate, tag_after_feat, in_train)
    length_feat = pos + '_ent_len_' + str(entity.end - entity.start)
    add_feature(candidate, length_feat, in_train)
    label_feat = pos + '_ent_label_' + entity.label_
    add_feature(candidate, label_feat, in_train)
    lemma_feat = pos + '_ent_lemma_' + entity.lemma_
    add_feature(candidate, lemma_feat, in_train)
    root_pos_feat = pos + '_root_pos_' + str(entity.end - entity.root.i)
    add_feature(candidate, root_pos_feat, in_train)
    root_dep_feat = pos + '_root_dep_' + entity.root.dep_
    add_feature(candidate, root_dep_feat, in_train)
    root_type_feat = pos + '_root_type_' + entity.root.ent_type_
    add_feature(candidate, root_type_feat, in_train)
    root_bracket_feat = pos + '_root_is-bracket_' + str(entity.root.is_bracket)
    add_feature(candidate, root_bracket_feat, in_train)
    root_digit_feat = pos + '_root_is-digit_' + str(entity.root.is_digit)
    add_feature(candidate, root_digit_feat, in_train)
    root_lower_feat = pos + '_root_is-lower_' + str(entity.root.is_lower)
    add_feature(candidate, root_lower_feat, in_train)
    root_head_dep = pos + '_root_head_dep_' + entity.root.head.dep_
    add_feature(candidate, root_head_dep, in_train)
    root_head_pos = pos + '_root_head_pos_' + entity.root.head.pos_
    add_feature(candidate, root_head_pos, in_train)
    root_head_tag = pos + '_root_head_tag_' + entity.root.head.tag_
    add_feature(candidate, root_head_tag, in_train)


'''
Adds the current candidates to the candidates list
'''
def add_current_candidates(candidate_distance, current_candidates, candidates):
    for c in current_candidates:
        if c in candidate_distance:
            c.apposition = True
        candidates.append(c)


'''
This function get a file name (text file), read the file, create for every sentence his candidates
 and for every candidate it extract his features
'''
def create_candidates(file_name, features, in_train):
    candidates = []
    global features_map
    features_map = features
    for sent_id, sent_str in read_lines(file_name):
        sent = nlp(sent_str)
        ents_len = len(sent.ents)
        candidate_distance = {}
        current_candidates = []
        for i in range(ents_len):
            for j in range(ents_len):
                if i != j:
                    entity1 = Entity(sent.ents[i], sent.ents[i].text)
                    entity2 = Entity(sent.ents[j], sent.ents[j].text)
                    if entity1.text.startswith('the'):
                        entity1.text = entity1.text[4:]
                    if entity2.text.startswith('the'):
                        entity2.text = entity2.text[4:]
                    candidate = Candidate(sent_id=sent_id, entity1=entity1, entity2=entity2)
                    create_environmental_features(sent, candidate, in_train, candidate_distance)
                    entities_combinations(candidate, i, j, in_train)
                    create_entities_features(sent, candidate, candidate.entity1.entity, 0, in_train)
                    create_entities_features(sent, candidate, candidate.entity2.entity, 1, in_train)
                    current_candidates.append(candidate)
        add_current_candidates(candidate_distance, current_candidates, candidates)
    if in_train:
        write_features_file(features_map)
    return candidates


'''
This function find for every candidate his real y
'''
def find_candidates_gold_y(candidates, annotated_fname):
    y = []
    annotations = load_annotated_file(annotated_fname, live_in=False)
    for candidate in candidates:
        current_y = 0
        sent_id = candidate.sent_id
        if sent_id in annotations:
            sent_annotations = annotations[sent_id]
            candidate_tuple = (candidate.entity1.text, candidate.entity2.text)
            if candidate_tuple in sent_annotations:
                candidate.set_label(1)
                current_y = 1
        y.append(current_y)
    return y


# train_tuples = create_candidates("Corpus.TRAIN.txt", {} ,in_train=True)
# csr_matrix = create_sparse_feature_vectors(train_tuples, features_map)
# train_y = find_candidates_gold_y(train_tuples, "TRAIN.annotations")
# train(csr_matrix,train_y)

