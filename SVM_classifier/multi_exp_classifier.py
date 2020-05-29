import SVMClassifier

class DispatcherModel():
  def __init__(self):
    self.model_class = SVMClassifier.SVMClassifier
    self.model_instances = {}
  
  def train_all(self, x, y):
    """ 
    @param x: 2D Table of length n with colummns: dialogname | expectation num | input sentence
    @param y: List of labels of length n, matching the lines of x
    """
    split_training_sets = {}
    
    for i in enumerate(x):
      label = y[i]
      dialog_name, exp_name, input_sentence = x[i]
      if expectation_num not in split_training_sets:
        split_training_sets[exp_name] = [[], []]
      split_training_sets[exp_name][0].append(input_sentence)
      split_training_sets[exp_name][1].append(label)
      
    for exp_name, (x_i, y_i) in split_training_sets.items():
                model = self_model_class()
                model.train(x_i, y_i)
                self.model_instances[exp_name] = model
  
  def predict(self, input_sentence, exp_name = None):
    if exp_name is None: 
      exp_names = self.model_instances.keys()
    else
      exp_names = [exp_name]
   results = {}
   for name, model in exp_names.items():
      label = model.predict(input_sentence)
      conf = model.confidence(input_sentence)
      results[name] = (label, conf)
   return results

   
   def loadDataset(self, file):
        dataset = pd.read_csv(file, encoding="latin-1")
        return dataset
     
    def get_params(self):
      pass
    def set_params(self, **params):
      pass
    
    def train(self, trainFeatures, Train_Y):
        self.model.fit(trainFeatures, Train_Y)
        print("Triaining complete")
        
    def predict(self, testFeatures):
        return self.model.predict(testFeatures)
    
    def score(self, model_predictions, Test_Y):
        return accuracy_score(model_predictions, Test_Y)*100
    
    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))
        print("Model saved successfully!")
        
    def load(self, filename):
        model = pickle.load(open(filename, 'rb'))
        return model

    def confidence_score(self, sentence, expectation_number):
        if expectation_number == None:
