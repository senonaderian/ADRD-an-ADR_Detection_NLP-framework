import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk
from memory_profiler import profile


def punctuation_removal(messy_str):
    if isinstance(messy_str, str):
        clean_list = [char for char in messy_str if char not in string.punctuation]
        clean_str = ''.join(clean_list)
        return clean_str
    else:
        return messy_str


def stopwords_removal(messy_str):
    if isinstance(messy_str, str):
        clean_list = [word for word in word_tokenize(messy_str) if word.lower() not in stopwords.words('english')]
        clean_str = ' '.join(clean_list)
        return clean_str
    else:
        return messy_str


def drop_numbers(messy_str):
    if isinstance(messy_str, str):
        list_text = word_tokenize(messy_str)
        clean_list = [word for word in list_text if not word.isnumeric()]
        clean_str = ' '.join(clean_list)
        return clean_str
    else:
        return str(messy_str)


def scale_rating(rating):
    rating -= min_rating
    rating = rating / (max_rating - 1)
    rating *= 5
    rating = int(round(rating, 0))

    if int(rating) == 0 or int(rating) == 1 or int(rating) == 2:
        return 0
    else:
        return 1

def read_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print("An error occurred while reading the dataset:", str(e))
        return None


@profile
def summarize_dataset(data):
    if data is None:
        return

    try:
        print("The Shape of the Dataset:", data.shape)
        print(data.head())
        print("Number of patient comments present in the Dataset:", data['drug_id'].nunique())
        print("Number of unique drugs in the Dataset:", data['drug_name'].nunique())
        print("Number of Unique Medical Conditions present in the Dataset:", data['condition'].nunique())
        print("Starting Date:", data[['date']][1:].max())
        print("Ending Date:", data[['date']][1:].min())
        print(data[['rating', 'ADR count']].describe())
        print("The Number of Drugs with No ADR Count:", data[data['ADR count'] == 0].count()[0])
        print("Number of patients who were satisfied using each drug (rating>=4) and experienced no ADR:", data[(data['ADR count'] == 0) & (data['rating'] >= 4)].count()[0])
        print("Average Rating of Drugs with No ADR Count: {0:.2f}".format(data[data['ADR count'] == 0]['rating'].mean()))
        print("Maximum ADR count extracted from comment column:", int(data['ADR count'].max()))
        print("Average Rating of Drugs with 20+ ADR Counts:", data[data['ADR count'] > 20]['rating'].mean())
        print("Name and Condition of these Drugs:")
        print(data[data['ADR count'] > 20][['drug_id', 'condition', 'dosage_duration', 'rating']].reset_index(drop=True))
        print(data[['drug_name', 'condition', 'comment', 'dosage_duration']].describe(include='object'))
        data = data.dropna()
        print(data.isnull().sum().sum())
    except Exception as e:
        print("An error occurred while summarizing the dataset:", str(e))

def visualize_distribution(data):
    try:
        plt.rcParams['figure.figsize'] = (15, 4)
        plt.subplot(1, 2, 1)
        sns.histplot(data['rating'])
        plt.subplot(1, 2, 2)
        sns.histplot(data['ADR count'])
        plt.suptitle('Distribution of Rating and ADR Count \n ', fontsize=20)
        plt.show()

        plt.rcParams['figure.figsize'] = (15, 4)
        sns.barplot(x=data['rating'], y=data['ADR count'], palette='hot')
        plt.grid()
        plt.xlabel('\n Ratings')
        plt.ylabel('Count\n', fontsize=20)
        plt.title('\n Rating vs ADR count \n', fontsize=20)
        plt.show()
    except Exception as e:
        print("An error occurred while visualizing the distribution:", str(e))

def clean_reviews(data):
    try:
        data['comment'] = data['comment'].apply(punctuation_removal)
        data['comment'] = data['comment'].apply(stopwords_removal)
        data['comment'] = data['comment'].apply(drop_numbers)
        return data
    except Exception as e:
        print("An error occurred while cleaning the reviews:", str(e))
        return None

def calculate_sentiment(data):
    try:
        sid = SentimentIntensityAnalyzer()
        data['sentiment'] = data['comment'].apply(lambda x: sid.polarity_scores(x)['compound'])
        data = data.drop(['date', 'sentiment', 'comment'], axis=1)
        if 'len' in data.columns:
            data = data.drop('len', axis=1)
        return data
    except Exception as e:
        print("An error occurred while calculating sentiment:", str(e))
        return None

def calculate_effectiveness(data):
    try:
        data['eff_score'] = data['rating'].apply(scale_rating)
        data['usefulness'] = data['rating'] * data['ADR count'] * data['eff_score']
        print(data[['drug_id', 'condition', 'usefulness', 'dosage_duration']][data['usefulness'] > data['usefulness'].mean()].sort_values(by='usefulness', ascending=False).head(10).reset_index(drop=True))
        return data
    except Exception as e:
        print("An error occurred while calculating effectiveness:", str(e))
        return None

def analyze_conditions(data):
    try:
        print("Number of Unique Conditions:", data['condition'].nunique())
        print(data['condition'].value_counts().head(10))
        data = data.drop_duplicates()
        return data
    except Exception as e:
        print("An error occurred while analyzing conditions:", str(e))
        return None

def high_low_rate(data, condition):
    try:
        print("\nTop 5 Useful Results")
        print(data[data['condition'] == condition][['drug_id', 'usefulness', 'dosage_duration']].sort_values(by='usefulness', ascending=False).head().reset_index(drop=True))

        print("\nBottom 5 Useful Results")
        print(data[data['condition'] == condition][['drug_id', 'usefulness', 'dosage_duration']].sort_values(by='usefulness', ascending=True).head().reset_index(drop=True))
    except Exception as e:
        print("An error occurred while analyzing high and low rate:", str(e))

# Set stopwords
stop = stopwords.words('english')
stop.append("i'm")
stop_words = [punctuation_removal(item) for item in stop]

# Read dataset
data = read_dataset("new 2.csv")

# Summarize dataset
summarize_dataset(data)

# Visualize distribution
visualize_distribution(data)

# Clean reviews
data = clean_reviews(data)

# Calculate sentiment
data = calculate_sentiment(data)

# Calculate effectiveness
if data is not None:
    min_rating = data['rating'].min()
    max_rating = data['rating'].max()
    data = calculate_effectiveness(data)

# Analyze conditions
if data is not None:
    data = analyze_conditions(data)

# Find most useful and useless drugs for each condition
if data is not None:
    condition_list = list(data['condition'].value_counts().index)
    high_low_rate(data, condition_list[0])
