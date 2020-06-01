from SVM_Class import SVMClassifier
from collections import defaultdict

class DispatcherModel():
  def __init__(self):
    self.model_obj = SVMClassifier()
    self.model_instances = {}
    self.score_dictionary = {}
    
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
      Train_X, Test_X, Train_Y, Test_Y = self.model_obj.split(processed_data,Train_Y)
      self.model_obj.initialize_ideal_answer(Train_X)
      Train_Y, Test_Y = self.model_obj.encode_y(Train_Y, Test_Y)
      features = self.model_obj.alignment(Train_X, None)
      C,kernel, degree, gamma, probability = self.model_obj.get_params()
      model = self.model_obj.set_params(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
      self.model_obj.train(features, Train_Y)
      self.model_obj.save('model_' + str(exp_num))
      self.model_instances[exp_num] = model

    return self.model_instances

  def predict_sentence(self, input_sentence, exp_num):

    sent_proc = self.model_obj.preprocessing(input_sentence)
    sent_features = self.model_obj.alignment(sent_proc, None)

    if exp_num is None:
      for exp_num,model in self.model_instances.items():
        model_score = self.model_obj.confidence_score(model, sent_features)
        model_class = self.model_obj.predict(model, sent_features)
        self.score_dictionary[exp_num] = [model_score, model_class[0]]
      
    else:
      model_score = self.model_obj.confidence_score(self.model_instances[exp_num], sent_features)
      model_class = self.model_obj.predict(self.model_instances[exp_num], sent_features)
      self.score_dictionary[exp_num] = [model_score, model_class[0]]
    return self.score_dictionary


obj = DispatcherModel()
model_instances = obj.train_all('comp_dataset.csv')

input_sentence = ['unfortunately entertain least Awful beyond belief!: I feel I have to write to keep others from wasting their money. This book seems to have been written by a 7th grader with poor grammatical skills for her age! As another reviewer points out, there is a misspelling on the cover, and I believe there is at least one per chapter. For example, it was mentioned twice that she had a "lean" on her house. I was so distracted by the poor writing and weak plot, that I decided to read with a pencil in hand to mark all of the horrible grammar and spelling. Please dont waste your money. I too, believe that the good reviews must have been written by the authors relatives. I will not put much faith in the reviews from now on!']
scores = obj.predict_sentence(input_sentence, None)
print(scores)

