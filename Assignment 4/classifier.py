from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


'''
This function get the train data and his y , train it with logistic regression algorithm
and save the trained model
'''
def train(train_data,train_y):
    model = LogisticRegression(C=50, penalty='l1')
    model.fit(train_data, train_y)
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)



'''
This function get the test data , load the trained model and return the prediction
'''
def predict(test_data):
    loaded_model = joblib.load('finalized_model.sav')
    predicted = loaded_model.predict(test_data)
    return predicted
