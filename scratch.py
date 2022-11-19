#imports
import sys
import pickle
import nltk
import multiprocessing
from joblib import parallel_backend
import pandas as pd
import numpy as np
import re
import string
import sqlite3
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    conn = sqlite3.connect(database_filepath)
    sqlquery = "SELECT * FROM \'" + database_filepath + "\'" 
    df = pd.read_sql(sqlquery,conn)
    
    X = df.message.values
    categories = ['related','request','offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    y = []
    for category in categories:
        y.append(df[category].values)
    y = np.transpose(np.array(y))
    return X, y, categories


#This function needs to: use a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. 
#This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.
def tokenize(text):

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
    return tokens

#The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. 
#GridSearchCV is used to find the best parameters for the model.
#The TF-IDF pipeline is only trained with the training data. 
def build_model():

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', AdaBoostClassifier(random_state=42))
    ])
    
    
    parameters = {
        'clf__n_estimators' : [25,50,75],
        'clf__learning_rate' : [0.75, 1.5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    multi_target_forest = MultiOutputClassifier(cv,n_jobs=-1)
    return multi_target_forest

#The f1 score, precision and recall for the test set is outputted for each category.
def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    labels = category_names
    #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", labels)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    #print("\nBest Parameters:", cv.best_params_)

def eval_model(Y_pred,Y_test):
    #init 36 len arrays to store each
    true_pos = np.zeros((36,))
    false_pos = np.zeros((36,))
    false_neg = np.zeros((36,))

    #loop through and add up each
    for array in range(len(Y_pred)):
        for entry in range(len(Y_pred[array])):
            pred_val = Y_pred[array][entry]
            true_val = Y_test[array][entry]
            
            #condition for true positive
            if pred_val == true_val:
                true_pos[entry] += 1 
            #condition for false pos
            elif (pred_val == 1) & (true_val == 0):
                false_pos[entry] += 1
            #condition for false neg
            elif (pred_val == 0) & (true_val == 1):
                false_neg[entry] += 1

    #loop through and define precision and recall for each category
    precision = np.zeros((36,))
    recall = np.zeros((36,))

    for i in range(len(precision)):
        precision[i] = true_pos[i] / (true_pos[i] + false_pos[i])
        recall[i] = true_pos[i] / (true_pos[i] + false_neg[i])

    f1 = np.zeros((36,))
    #calculate f1 for each
    for i in range(len(f1)):
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    #store results in dataframe
    results = pd.DataFrame()
    results["Category"] = categories
    results["f1"] = f1
    results["precision"] = precision
    results["recall"] = recall

    #return dataframe
    return results

def main():
    print("loading data...")
    X, Y, category_names = load_data("data/DisasterResponse.db")
    print("splitting data...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print("building model...")
    model=build_model()
    print("training model...")
    model.fit(X_train,Y_train)
    print("predicting on test data...")
    Y_pred = model.predict(X_test)
    print("scoring data...")
    print(model.score(X_test,Y_test))
    print("saving resultant dataframe to results.csv")
    pd.to_csv(eval_model(Y_pred,Y_test),"results.csv")

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    parallel_backend("threading")
    main()