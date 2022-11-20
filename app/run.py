import json
import plotly
import re
import pandas as pd
import nltk

import joblib

from nltk.stem import WordNetLemmatizer
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords','omw-1.4'])


app = Flask(__name__)

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

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

# load models
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    sum_names = df.columns[4:]
    sum_counts = {}
    for col in sum_names:
        sum_counts[col] = df[col].sum()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=sum_names,
                    y=list(sum_counts.values())
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classification',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification",
                    'tickangle' : 45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': "Genres",
                'yaxis': {
                    'title': "Genre Counts"
                },
                'xaxis': {
                    'title': "Genre Names"
                }
                }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query
    
    classification_labels = model.predict([query])
    classification_results = dict(zip(df.columns[4:].tolist(), classification_labels.T))
        



    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()