from SVM_Class import SVMClassifier
from collections import defaultdict

class DispatcherModel():
  def __init__(self):
    self.model_obj = SVMClassifier()
    self.model_instances = {}
    self.score_dictionary = {}
    self.ideal_answers_dictionary = {}

  def train_all(self, filename):

    split_training_sets= defaultdict(int)

    Corpus = self.model_obj.loadDataset(filename)
    
    for i,value in enumerate(Corpus['exp_num']):
      if value not in split_training_sets:
          split_training_sets[value] = [[], []]
      split_training_sets[value][0].append(Corpus['text'][i])
      split_training_sets[value][1].append(Corpus['label'][i])

    for exp_num, (Train_X, Train_Y) in split_training_sets.items():
      processed_data = self.model_obj.preprocessing(Train_X)
      ia = self.model_obj.initialize_ideal_answer(processed_data)
      self.ideal_answers_dictionary[exp_num] = ia
      Train_X, Test_X, Train_Y, Test_Y = self.model_obj.split(processed_data,Train_Y)
      
      Train_Y, Test_Y = self.model_obj.encode_y(Train_Y, Test_Y)
      features = self.model_obj.alignment(Train_X, None)
      C,kernel, degree, gamma, probability = self.model_obj.get_params()
      model = self.model_obj.set_params(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
      self.model_obj.train(features, Train_Y)
      self.model_instances[exp_num] = model
    self.model_obj.save(self.model_instances, 'models')
    self.model_obj.save(self.ideal_answers_dictionary, 'ideal_answers')


  def load(self,models, ideal_answers):
    return self.model_obj.load('models'), self.model_obj.load('ideal_answers')

  def predict_sentence(self, input_sentence, exp_num):

    sent_proc = self.model_obj.preprocessing(input_sentence)
    self.model_instances, self.ideal_answers_dictionary = self.load('models', 'ideal_answers')
    
    if exp_num is None:
      for exp_num,model in self.model_instances.items():
        sent_features = self.model_obj.alignment(sent_proc,self.ideal_answers_dictionary[exp_num])
        model_score = self.model_obj.confidence_score(model, sent_features)
        model_class = self.model_obj.predict(model, sent_features)
        if model_class[0] == 0:
          class_name = "Good" 
        else:
          class_name = "Bad" 
        self.score_dictionary[exp_num] = [model_score, class_name]
      
    else:
      sent_features = self.model_obj.alignment(sent_proc,self.ideal_answers_dictionary[exp_num])
      model_score = self.model_obj.confidence_score(self.model_instances[exp_num], sent_features)
      model_class = self.model_obj.predict(self.model_instances[exp_num], sent_features)
      if model_class[0] == 0:
        class_name = "Good" 
      else:
        class_name = "Bad" 
      self.score_dictionary[exp_num] = [model_score, class_name]
    return self.score_dictionary

if __name__ == '__main__':
  obj = DispatcherModel()
  model_instances = obj.train_all('comp_dataset.csv')

  input_sentence = ['rules can make you unpopular'] 
  scores = obj.predict_sentence(input_sentence, None)
  print("scores=  ", scores)

