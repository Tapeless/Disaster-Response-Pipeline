import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import pandas as pd
import numpy as np
import re
import string
import sqlite3
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import MultiOutputClassifier


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

def tokenize(text):
    '''
    tokens = []
    for entry in text:
        entry = re.sub(r"[^a-zA-Z0-9]", " ", entry.lower())
        tokens.append(word_tokenize(entry))
     
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for item in tokens:
        new_row = []
        for word in item:
            clean_tok = lemmatizer.lemmatize(word).lower().strip()
            new_row.append(clean_tok)
            clean_tokens.append(new_row)
    return clean_tokens    
    '''
def vectorize(text):
    # initialize tf-idf vectorizer object
    #text = tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english',use_idf=False)
    X = vectorizer.fit_transform(text)
    
    #export vectorizer as pickle
    pickle.dump(vectorizer, open("models/vectorizer.pkl", 'wb'))

    return X

def split_data(X, Y, test_size=0.2):
    X = vectorize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test

def build_model():

    pipeline = Pipeline([
        ('clf', AdaBoostClassifier(random_state=42,n_estimators=125,learning_rate=1.5))
    ])
    
    '''
    parameters = {
        'clf__n_estimators' : [10,25,50],
        'clf__max_leaf_nodes' : [None, 50]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    '''
    
    multi_target_forest = MultiOutputClassifier(pipeline,n_jobs=1)
    return multi_target_forest


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    labels = category_names
    #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", labels)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    #print("\nBest Parameters:", cv.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()