import .SVMClassifier

class DispatcherModel():
  def __init__(self):
    self.model_class = SVMClassifier
    self.model_instances = {}
  
  def train_all(self, x, y):
         splitTrainingSets = (split into a dictionary of training data sets)
         for i, x_i, y_i in splitTrainingSets:
                model = self._model_class()
                model.train(x_i, y_i)
                self.model_instances[i] = model
