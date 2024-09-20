# main.py
import pandas as pd
import os
import re
from sklearn.metrics import classification_report
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Define the directory paths for the dataset
train_dir = '../data/aclImdb/train'  # Adjust path as per your file structure
test_dir = '../data/aclImdb/test'


def load_data_from_folder(folder):
    """
    Load reviews from the specified folder (either 'train' or 'test').
    
    Args:
    folder (str): Path to the folder containing 'pos' and 'neg' subfolders.

    Returns:
    DataFrame: A DataFrame containing reviews and their corresponding sentiment labels.
    """
    reviews = []
    sentiments = []
    
    # Iterate over 'pos' and 'neg' subfolders
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(folder, sentiment)
        
        # Load each review text file
        for filename in os.listdir(sentiment_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(sentiment_dir, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    reviews.append(review)
                    # Add sentiment label (1 for positive, 0 for negative)
                    sentiments.append(1 if sentiment == 'pos' else 0)

    return pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })


# Load training and test data
train_data = load_data_from_folder(train_dir)
test_data = load_data_from_folder(test_dir)

# Combine train and test data (optional step)
combined_data = pd.concat([train_data, test_data])


# Load spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    """
    Preprocess a given text (lowercasing, removing punctuation, stopwords, lemmatization).

    Args:
    text (str): The raw text of a review.

    Returns:
    str: Preprocessed text.
    """
    text = text.lower()  # Step 1: Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Step 2: Remove punctuation
    
    # Step 3: Remove stopwords and punctuation using spaCy
    doc = nlp(text)
    text = ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])
    
    # Step 4: Lemmatization
    text = ' '.join([token.lemma_ for token in doc])
    
    # Step 5: Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Step 6: Remove extra whitespaces
    return ' '.join(text.split())


# Apply preprocessing to the reviews
combined_data['preprocessed_review'] = combined_data['review'].apply(preprocess_text)

# Save preprocessed data
combined_data.to_csv("../preprocessed.csv", index=False)

# Load preprocessed data
df = pd.read_csv("../preprocessed.csv")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.preprocessed_review, df.sentiment, test_size=0.2, random_state=2024)

# Initialize CountVectorizer for converting text to vectors
v = CountVectorizer()

# Transform training data into vectors
x_train_cv = v.fit_transform(x_train)

# Train a Naive Bayes classifier
model = MultinomialNB().fit(x_train_cv, y_train)

# Transform test data into vectors using the same vectorizer
x_test_cv = v.transform(x_test)

# Make predictions on the test set
y_pred = model.predict(x_test_cv)

# Print the classification report
print(classification_report(y_test, y_pred))
