import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories datasets, merges them using the common id.

    Args:
        messages_filepath (str): Path to the messages dataset (csv file).
        categories_filepath (str): Path to the categories dataset (csv file).

    Returns:
        pd.DataFrame: The return value, which merged the two datasets messages and categories.

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """Cleans the given Pandas DataFrame.

    Args:
        df (pd.DataFrame): Pandas DataFrame object to be cleaned.

    Returns:
        pd.DataFrame: The return value, which went through the cleaning process.

    """

    # Split categories into separate category columns.
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        # print out non-binary column
        if not np.isin(categories[column].unique(), [0, 1]).all():
            print('category {} is not binary.\n{}'.format(column, categories[column].value_counts()))

    # Replace categories column in df with new category columns.
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Cleaning for related category
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    print('category {} is now binary.\n{}'.format('related', df['related'].value_counts()))

    # Remove duplicates.
    if df.duplicated().sum() != 0:
        df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Saves Pandas DataFrame to SQL database.

    Args:
        df (pd.DataFrame): DataFrame object to save to `database_filename`.
        database_filename (str): Path to save SQL database.

    Returns:
        None

    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_categories', engine, if_exists='replace', index=False)


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
