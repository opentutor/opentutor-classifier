from opentutor_classifier.svm import SVMAnswerClassifierTraining
from opentutor_classifier import loadData
import os
data_path = os.path.join(os.getcwd(), 'data')
print(data_path)
data = loadData(os.path.join(data_path, 'training_data.csv'))
print(type(data))
obj_train = SVMAnswerClassifierTraining()
obj_train.train_all(corpus=data, output_dir="models")
