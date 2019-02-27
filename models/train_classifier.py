import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import re
import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score

import time
from pprint import pprint
import joblib
######

def load_data(database_filepath):
    """Loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    # table name
    table_name = 'labeled_messages'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    
    
    X = df['message']
    Y = df.iloc[:,4:]
    #OR
    #Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    global category_names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


#def tokenize(text):
#    pass

def clean_text(text, ):
    """ A series of nested functions to preprocess the text data
        Functions: 
            Tokenize Text 
            remove special characters
            lemmatize_text
            remove_stopwords
        Returns:
            Clean and preprocessed text 
    """    
    
    # Set checking is faster in Python than list.
    default_stopwords = set(stopwords.words("english"))  
    default_lemmatizer = WordNetLemmatizer()
    # Search for all non-letters and replace with spaces
    text = re.sub("[^a-zA-Z]"," ", str(text))    
        
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]
    
    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def lemmatize_text(text, lemmatizer=default_lemmatizer):
        tokens = tokenize_text(text)
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)
    
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = remove_special_characters(text) # remove punctuation and symbols
    text = lemmatize_text(text) # stemming
    text = remove_stopwords(text) # remove stopwords

    return text

def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=clean_text)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    #Uncomment for additional parameters
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
              #'vect__max_df': (0.5, 0.75, 1.0),
              #'vect__max_features': (None, 5000, 10000),
              'tfidf__use_idf': (True, False),
              #'clf__estimator__min_samples_split': [2, 4],
              'clf__estimator__n_estimators': [10, 50, 100]}
    #Create Model
    cv = GridSearchCV(pipeline, parameters, cv=3,
                           n_jobs=-1, verbose=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    # Generate predictions
    Y_pred = model.predict(X_test)

    # Print out the full classification report and accuracy score
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('---------------------------------')
    #global categories
    for i in range(Y_test.shape[1]):
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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