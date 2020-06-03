from opentutor_classifier.dispatcher_model import DispatcherModel
import os

data = os.path.join(os.getcwd(), 'data')
print(f"data={data}")
obj = DispatcherModel()
model_instances = obj.train_all(os.path.join(data, "comp_dataset.csv"))
input_sentence = ["rules can make you unpopular"]
scores = obj.predict_sentence(input_sentence, None)
print("scores=  ", scores)