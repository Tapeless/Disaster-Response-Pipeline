# Disaster Response Pipeline
![image](https://user-images.githubusercontent.com/72606788/202878115-e81affd3-c8ac-45eb-8f85-d53cfac85e4f.png)
This repo contains python scripts to clean data and generate an NLP model trained on the figure8 disaster response dataset available through Udacity Data Science Nanodegree program.
This dataset contains ~30000 pre-labeled tweets related (and not related!) to a flood.
After training and scoring the model, it is saved to a pickle file and utilized in a web dashboard.

### Required Packages:
- pandas
- numpy
- sqlite3
- sqlalchemy
- sklearn
- pickle
- nltk
- plotly
- joblib
- flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/ to view the webpage. 
