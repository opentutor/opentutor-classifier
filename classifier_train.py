from opentutor_classifier.multi_exp_classifier import DispatcherModel

obj = DispatcherModel()
model_instances = obj.train_all("comp_dataset.csv")
input_sentence = ["rules can make you unpopular"]
scores = obj.predict_sentence(input_sentence, None)
print("scores=  ", scores)