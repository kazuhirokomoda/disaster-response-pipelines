import sys
import pandas as pd
import numpy as np
import nltk
import pickle
import re

# download nltk libraries
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from custom_transformer import StartingVerbExtractor


URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """Loads dataset, returns feature and target values for machine learning.

    Args:
        database_filepath (str): Path to the SQL database (db file).

    Returns:
        np.ndarray: Feature variable X (array of str).
        np.ndarray: Target variable Y (array of 0 and 1).
        np.ndarray: Array of category names.

    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_categories', engine)
 
    X = df.message.values

    # DataFrame whose columns except for categories are removed
    df_categories = df.drop(columns=['id', 'message', 'original', 'genre'])

    Y = df_categories.values
    category_names = np.array(df_categories.columns)

    return X, Y, category_names


def tokenize(text):
    """Performs tokenization processes to input text to return list of
    processed words.

    Args:
        text (str): Text to be processed.

    Returns:
        list: List of processed words.

    """

    # Replace all urls in text with the str 'urlplaceholder'
    detected_urls = re.findall(URL_REGEX, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalization: Lowercase conversion and punctuation removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # Tokenization using nltk
    tokens = word_tokenize(text)

    # Stop Word Removal: Remove uninformative words
    words = [t for t in tokens if t not in stopwords.words("english")]

    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    # Reduce lemmatized words to their stems
    stemmed = [PorterStemmer().stem(l) for l in lemmed]

    return stemmed


def build_model():
    """Pipeline to vectorize and then apply TF-IDF to the text.

    Args:
        None.

    Returns:
        Pipeline or GridSearchCV object: scikit-learn pipeline of transforms 
        with a final estimator.

    """

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs=-1)))
    ])
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs=-1)))
    ])

    # TODO: grid search not ending forever
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1),), #((1, 1), (1, 2), (2, 2)), # default (1, 1)
        'features__text_pipeline__vect__max_df': (0.5, 1.0), #(0.5, 0.75, 1.0), # default 1.0
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000), # default None
        #'features__text_pipeline__tfidf__use_idf': (True, False), # default True
        #'clf__estimator__n_estimators': (50, 100, 200), # default 100
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    #return pipeline
    return cv


def get_scores(y_true, y_pred):
    """

    """

    # The f1 score, precision and recall for the test set.
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f_1 = f1_score(y_true, y_pred, average='micro')

    scores = {
        'Precision': precision,
        'Recall': recall,
        'F-1': f_1
    }
    return scores


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model with the test data and shows result.

    Args:
        model (Pipeline or GridSearchCV object): scikit-learn pipeline.
        X_test (np.ndarray): .
        Y_test (np.ndarray): .
        category_names (np.ndarray): .

    Returns:
        None

    """

    # Print best parameters if GridSearchCV is used
    if hasattr(model, 'best_params_'):
        print(model.best_params_)


    Y_pred = model.predict(X_test)
    df = pd.DataFrame(
        [get_scores(Y_test[:, idx], Y_pred[:, idx]) for idx in range(Y_test.shape[-1])],
        index=category_names
    )
    print(df)


def save_model(model, model_filepath):
    """Stores the classifier into a pickle file to the specified model file path.

    Args:
        model (Pipeline or GridSearchCV object): scikit-learn pipeline.
        model_filepath (str): File path to save the model.

    Returns:
        None

    """

    pickle.dump(model, open(model_filepath, 'wb'))


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
