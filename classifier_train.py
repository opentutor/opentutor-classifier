from opentutor_classifier.dispatcher_model import DispatcherModel
import os

data_path = os.path.join(os.getcwd(), 'data')
obj = DispatcherModel()
data = obj.loadData(os.path.join(data_path, "comp_dataset.csv"))
model_instances = obj.train_all(data)
input_sentence = ["rules can make you unpopular"]
scores = obj.predict_sentence(input_sentence, None)
print("scores=  ", scores)
