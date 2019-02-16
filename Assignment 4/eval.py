from utils import load_annotated_file
import sys


'''
This function get the gold annotations and the predicted annotations and return the precision
'''
def calculate_precision(gold_annotations,prediction_annotations):
    good = bad = 0.0
    for sent_id, relations in prediction_annotations.iteritems():
        for relation in relations:
            if sent_id not in gold_annotations:
                bad += 1
            elif relation not in gold_annotations[sent_id]:
                bad += 1
            else:
                good += 1

    return good/(good+bad)


'''
This function get the gold annotations and the predicted annotations and return the recall
'''
def calculate_recall(gold_annotations,prediction_annotations):
    good = bad = 0.0
    for sent_id, relations in gold_annotations.iteritems():
        for relation in relations:
            if sent_id not in prediction_annotations:
                bad += 1
            elif relation not in prediction_annotations[sent_id]:
                bad += 1
            else:
                good += 1

    return good / (good + bad)


def main():
    # get the inputs
    gold_file_name = sys.argv[1]
    prediction_file_name = sys.argv[2]

    gold_annotations = load_annotated_file(gold_file_name,live_in=False)
    prediction_annotations = load_annotated_file(prediction_file_name,live_in=False)

    precision = calculate_precision(gold_annotations,prediction_annotations)
    recall = calculate_recall(gold_annotations,prediction_annotations)

    f1 = 2*((precision*recall)/(precision+recall))

    print 'Precision = ' +str(precision)
    print 'Recall = ' + str(recall)
    print 'F1 = ' + str(f1)

if __name__ == "__main__":
    main()
