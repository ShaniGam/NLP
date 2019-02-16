import sys
from utils import write_annotated_file, load_features_file
from classifier import predict
from feature_extractor import create_candidates
from csr_converter import create_sparse_feature_vectors


'''
This function check that the input file is in txt format
'''
def check_file_format(fname):
    for line in file(fname):
        first = line.strip()
        break
    if first[0] == '#':
        print 'Wrong file format, please make sure that your input file format is filename.txt '
        sys.exit()


def main():
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    check_file_format(input_file_name)
    feats_map = load_features_file()
    test_candidates = create_candidates(input_file_name, feats_map,in_train=False)
    csr_test = create_sparse_feature_vectors(test_candidates, feats_map)
    predictions = predict(csr_test)
    write_annotated_file(test_candidates, predictions, output_file_name, live_in=False)


if __name__ == "__main__":
    main()
