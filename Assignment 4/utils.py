import pickle


'''
This function load the annotated data for the relation Work_for (live_in = false)
'''
def load_annotated_file(fname, live_in):
    annotations_dict = {}
    for line in file(fname):
        list = line.strip().split('\t')
        if len(list) > 4:
            sent_id, arg1, rel, arg2 = list[:-1]
        else:
            sent_id, arg1, rel, arg2 = list
        if arg1.endswith('.'):
            arg1 = arg1[:-1]
        if arg2.endswith('.'):
            arg2 = arg2[:-1]
        if (live_in == True and rel == 'Live_In') or (live_in == False and  rel == 'Work_For'):
            if sent_id in annotations_dict:
                annotations_dict[sent_id].append((arg1, arg2))
            else:
                annotations_dict[sent_id] = [(arg1, arg2)]
    return annotations_dict


'''
This function write the annotated file for the candidates list , their predictions and the relation Work_for (live_in = false)
'''
def write_annotated_file(candidates_list,predictions,output_fname,live_in):
    out_str = ''
    for i in range(len(candidates_list)):
        has_appos = predictions[i] == 0 and candidates_list[i].apposition
        if predictions[i] == 1 or has_appos :
            if live_in:
                rel = 'Live_In'
            else:
                rel = 'Work_For'
            sent_id = candidates_list[i].sent_id
            entity1 = candidates_list[i].entity1.text
            entity2 = candidates_list[i].entity2.text
            out_str += '\t'.join([sent_id,entity1,rel,entity2]) + '\n'

    out = file(output_fname,'w')
    out.write(out_str)
    out.close()


'''
This function save the feature map (from the training) into a file
'''
def write_features_file(features_map):
    pickle.dump(features_map, open("features_map", "wb"))


'''
This function load the features map (for using it in the prediction
'''
def load_features_file():
    feats_mappingg = pickle.load(open("features_map", "rb"))
    return feats_mappingg
