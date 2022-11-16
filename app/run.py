import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def tokenize(text):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(text)

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load models
model = joblib.load("../models/classifier.pkl")
vect_model = joblib.load("../models/vectorizer.pkl")


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
                    'title': "Classification"
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
    
    classification_labels = model.predict(vect_model.transform([query]))
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