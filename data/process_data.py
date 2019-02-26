import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loads:
    The specified message and category data
    
    Args:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories cv

    Returns:
        df (pandas dataframe): Merged messages and categories df, merged on ID
    """
    #import data from cv
    messages = pd.read_csv(messages_filepath)
    global categories
    categories = pd.read_csv(categories_filepath)
    
    #merge Data on ID
    #this could also work in a pinch
    #df = messages.merge(categories, on='id', how='left')
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """Cleans the data:
        - splits categories into separate columns
        - converts categories values to binary values
        - drops duplicates
    
    Args:
        df (pandas dataframe): combined categories and messages df
    Returns:
        df (pandas dataframe): Cleaned dataframe with split categories
    """
    global categories
    # expand the categories column
    categories = categories.categories.str.split(';', expand=True)
    
    
    # Extract the category names
    row = categories[:1]
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames

    # get only the last value in each value as an integer
    categories = categories.applymap(lambda x: int(x[-1]))
    
    #Transform non binary values (2) to zero
    categories.related.replace(2, 0, inplace=True)
    
    # concatenate the categories back to the original df
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    #Remove all rows with empty category columns
    df.dropna(subset= ['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report'], inplace = True)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df



def save_data(df, database_filename):
    """Saves the preprocessed data to a sqlite db
    Args:
        df (pandas dataframe): The cleaned dataframe
        database_filename (string): the file path to save the db
    Returns:
        None
    """
    #Save the clean dataset into an sqlite database
    table_name = 'labeled_messages'
    # create engine 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists 
    df.to_sql(table_name, engine, index=False, if_exists='replace')
   
    #If in need of disposal, uncomment below
    #https://docs.sqlalchemy.org/en/latest/core/connections.html
    #engine.dispose()



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