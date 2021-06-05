# Disaster Response Pipelines

## Project Motivation

To utilize the [real life disaster datasets](https://github.com/kazuhirokomoda/disaster-response-pipelines/tree/master/data) provided by [Figure Eight](https://appen.com/) to perform the followings:

- Build an ETL Pipeline to clean the data.
- Build a supervised learning model to categorize messages, using scikit-learn `Pipeline` and `FeatureUnion`.
- Build a Flask web app to:
  - Take an user input message and get the classification results for 36 categories (ex. `aid_related`, `weather_related`, `direct_report`, etc.).
  - Display visualizations to describe the datasets.

## Installation

### Dependencies

In addition to the [Anaconda distribution of Python (versions 3.\*)](https://www.anaconda.com/products/individual-b), following libraries need to be installed manually.

- [plotly](https://plotly.com/python/)

### How to run the project

1. Navigate yourself to the project's root directory.

2. To run ETL pipeline that cleans data and stores in database:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

3. To run ML pipeline that trains classifier and saves it:

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

4. To run the web app:

```
cd app; pwd
python run.py
```

5. Finally, go to http://0.0.0.0:3001/ to find visualizations describing the dataset. You can also input text message to see which categories it falls into, according to the trained model's prediction.

## File Descriptions

- `data/` directory includes ETL script `process_data.py` to clean data and stores in a SQLite database
- `models/` directory includes ML script `train_classifier.py` to train a classifier and store it into a pickle file
- `app/` directory includes web app `run.py` to handle visualizations using data from SQLite database

## Licensing, Authors, Acknowledgements

- [Figure Eight](https://appen.com/) for preparing the datasets
- [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
