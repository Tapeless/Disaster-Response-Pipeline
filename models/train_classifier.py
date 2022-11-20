import sys
import pickle
import nltk
import pandas as pd
import numpy as np
import re
import sqlite3
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

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
    Takes in strings of text and tokenizes/lemmatizes.
    Called in the pipeline by TfIdfVectorizer.
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
    return tokens

def build_model():
    '''
    Defines a pipeline to tokenize text input and classify categories.
    GridSearchCV is used to find most performant parameters from options in parameters dict
    '''

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    
    
    parameters = {
        'clf__estimator__n_estimators' : [25,50,100],
        'clf__estimator__learning_rate' : [0.75, 1.5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1)
    return cv

def eval_model(Y_pred,Y_test,categories):
    '''
    This function takes in the values predicted + ground truth values, and calculates the
    precision, recall, and f1 score for each category.
    This is necessary because sklearn's scoring does not support multioutput classification.
    A dataframe with the precision/recall/f1 for each category is returned.
    '''
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
    
    #print results to terminal
    #22 is longest category string length
    fmt = '{:<22} {:<8} {:<6} {:<8}'
    print(fmt.format("Categories","Precision","Recall","f1 score")))
    for i in range(len(f1)):
        print(fmt.format(np.round(categories[i],3),np.round(precision[i],3),np.round(recall[i],3),np.round(f1[i],3)))

    #store results in dataframe
    results = pd.DataFrame(list(zip(categories,precision,recall,f1)),columns=["Categories","Precision","Recall","f1_score"])

    #return dataframe
    return results

def save_model(model, model_filepath):
    '''
    Saves trained model to pickle file.
    Args:
    model - model to save
    model_filepath - filepath to save to
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(f"Best Params: {model.best_params_}")

        print('Evaluating model...')
        Y_pred = model.predict(X_test)
        eval_model(Y_pred, Y_test, category_names)

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