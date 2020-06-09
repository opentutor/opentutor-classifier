from opentutor_classifier.svm import SVMAnswerClassifierTraining, SVMAnswerClassifier, AnswerClassifierInput, load_instances
from opentutor_classifier import loadData
import os
data_path = os.path.join(os.getcwd(), 'data')
data = loadData(os.path.join(data_path, 'training_data.csv'))
# print(type(data))
obj_train = SVMAnswerClassifierTraining()
output_dir = os.path.join(os.getcwd(), 'tests', 'fixtures', 'models')
obj_train.train_all(corpus=data, output_dir=output_dir)
model_instances, ideal_answers = load_instances(output_dir, 'model_instances', 'ideal_answers')
obj_test = SVMAnswerClassifier(model_instances, ideal_answers)
input_sentence = ["peer pressure"]
answer = obj_test.evaluate(AnswerClassifierInput(input_sentence=input_sentence, expectation=-1))
# print("answer = ", answer)
