# # Functions working:
# -     loadDataset: load the dataset 
# -     preprocessing: convert into lower cases + tokenization + lemmatization
# -     split: split dataset into 75% training and 25% testing.
# -     initialize_ideal_answer: define ideal_answer
# -     encode_y = coverting y lables(categorical) to numbers(0/1)
# -     word_overlap_score: find similarity by comparing Training examples with an ideal answer
# -     alignment: returns features, which includes scores for each training examples.
# -     get_params: define parameters for the model.
# -     set_params: set the parameters of the model.
# -     trian: train the model
# -     predict: performance on the testing dataset.
# -     score: finding accuracy
# -     save: save the model locally
# -     load: load the model from local
# -     predict_probabilities: find confidence scores.
# 
# Note: Dimension of the dataset is 100x2 (Its a matrix). The alignment function is made such that it will return a list (2d array)(Its a list). This is useful as we can use the same function for extracting features of training examples, testing examples, and for a new sentence .




import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle
from collections import defaultdict


np.random.seed(1)
# will remove in the development mode.


class SVMClassifier:
    def __init__(self):
        self.tag_map = defaultdict(lambda : wn.NOUN)
        self.tag_map['J'] = wn.ADJ
        self.tag_map['V'] = wn.VERB
        self.tag_map['R'] = wn.ADV
        self.ideal_answer = None
        self.model = None
        self.score_dictionary = defaultdict(int)
        
    def loadDataset(self, file):
        dataset = pd.read_csv(file, encoding="latin-1")
        return dataset
    
    def preprocessing(self, data):
        preProcessedDataset = []
        data = [entry.lower() for entry in data]
        data = [word_tokenize(entry) for entry in data]
        for index,entry in enumerate(data):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,self.tag_map[tag[0]])
                    Final_words.append(word_Final)
            preProcessedDataset.append(Final_words)
        return preProcessedDataset
    
    def split(self, preProcessedDataset,target):
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(preProcessedDataset,target,test_size=0.25)
        return Train_X, Test_X, Train_Y, Test_Y
    
    def initialize_ideal_answer(self, X):
        self.ideal_answer = X[0]
        return self.ideal_answer

    def encode_y(self, Train_Y, Test_Y):
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)
        return Train_Y, Test_Y
    
    def word_overlap_score(self, Train_X, ideal_answer):
        features = []
        for example in Train_X:
            intersection = set(ideal_answer).intersection(set(example)) 
            score = len(intersection)/len(set(ideal_answer))
            features.append(score)
        return features
        
    #function for extracting features
    def alignment(self, Train_X, ideal_answer):
        if ideal_answer is None:
            ideal_answer = self.ideal_answer
        features = self.word_overlap_score(Train_X, ideal_answer)
        return (np.array(features)).reshape(-1,1)
    
    def get_params(self):
        C=1.0
        kernel='linear'
        degree=3
        gamma='auto'
        probability=True
        return C,kernel, degree, gamma, probability
    
    def set_params(self, **params):
        self.model = svm.SVC(C = params['C'], kernel = params['kernel'], degree = params['degree'], gamma = params['gamma'], probability = params['probability'] )
        return self.model
    
    def train(self, trainFeatures, Train_Y):
        self.model.fit(trainFeatures, Train_Y)
        # print("Triaining complete")
        
    def predict(self, model, testFeatures):
        return model.predict(testFeatures)
     
    def score(self, model_predictions, Test_Y):
        return accuracy_score(model_predictions, Test_Y)*100
    
    def save(self, models, filename):
        pickle.dump(models , open(filename, 'wb'))
        # print("Model saved successfully!")
        
    def load(self, filename):
        model = pickle.load(open(filename, 'rb'))
        return model

    def confidence_score(self, model, sentence):
        return model.decision_function(sentence)[0]

if __name__ == '__main__':
    expectation1 = SVMClassifier()
    Corpus = expectation1.loadDataset('comp_dataset.csv')
    # print(Corpus)

    preProcessedDataset = expectation1.preprocessing(Corpus['text'])
    Train_X, Test_X, Train_Y, Test_Y = expectation1.split(preProcessedDataset, Corpus['label'])
    expectation1.initialize_ideal_answer(Train_X)
    Train_Y, Test_Y = expectation1.encode_y(Train_Y, Test_Y)
    features = expectation1.alignment(Train_X, None)

    C,kernel, degree, gamma, probability = expectation1.get_params()
    model = expectation1.set_params(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
    expectation1.train(features, Train_Y)
    train_pred = expectation1.predict(model, features)

    testFeatures = expectation1.alignment(Test_X, None)
    model_predictions = expectation1.predict(model, testFeatures)
    accuracy = expectation1.score(model_predictions, Test_Y)
    # print("Accuracy of the model: ",accuracy)

    expectation1.save(model, 'model1')

    model1 = expectation1.load('model1')

    sentence = ['Peer pressure can cause you to allow inappropriate behavior.']
    sent_proc = expectation1.preprocessing(sentence)
    sent_features = expectation1.alignment(sent_proc, None)

    score = expectation1.confidence_score(model1, sent_features)
    # print("score=  ", score)