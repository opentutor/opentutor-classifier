from opentutor_classifier.svm import SVMAnswerClassifier
from opentutor_classifier import AnswerClassifierInput, loadData
import os

data_path = os.path.join(os.getcwd(), 'data')
obj = SVMAnswerClassifier()
data = loadData(os.path.join(data_path, "comp_dataset.csv"))
obj.train_all(data)
input_sentence = ["peer pressure"]
result = obj.evaluate(AnswerClassifierInput(input_sentence=input_sentence, expectation=None))
print("len of result = ", len(result.expectationResults))
