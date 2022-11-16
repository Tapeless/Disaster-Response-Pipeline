import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories,on='id')


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    
    # select the first row of the categories dataframe to extract new name list
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert categories to one-hot
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #drop original categories column
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join='inner')
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
    

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()