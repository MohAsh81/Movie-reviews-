# Movie-reviews-
# IMDb Sentiment Analysis

This project uses the IMDb movie review dataset for binary sentiment analysis (positive/negative reviews). We use Naive Bayes as the classification model and perform text preprocessing (tokenization, lemmatization, etc.) to prepare the data for analysis.

## Dataset

You will need the `aclImdb` dataset, which contains the `train` and `test` directories. Each directory has two subdirectories: `pos` (positive reviews) and `neg` (negative reviews). Place the dataset inside the `data/` folder.

## Preprocessing

We preprocess the text data using the following steps:

- Lowercasing
- Removing punctuation and numbers
- Removing stopwords
- Lemmatizing words
