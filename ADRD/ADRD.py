#Importing the Libraries

import numpy as np
import pandas as pd
import string

import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets
from ipywidgets import interact

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk


def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str

def stopwords_removal(messy_str):
    messy_str = word_tokenize(messy_str)
    return [word.lower() for word in messy_str 
            if word.lower() not in stop_words ]

def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)

def scale_rating(rating):
    rating -= min_rating
    rating = rating/(max_rating -1)
    rating *= 5
    rating = int(round(rating,0))
    
    if(int(rating) == 0 or int(rating)==1 or int(rating)==2):
        return 0
    else:
        return 1




plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('fivethirtyeight')

#Phase1: Reading the Dataset
data = pd.read_csv(r"new 2.csv")
#data = pd.read_csv(r"lexaproCSV.csv")
#data = pd.read_csv(r"ZoloftCSV.csv")
#data = pd.read_csv(r"CymbaltaCSV.csv")
#data = pd.read_csv(r"EffexorXRCSV.csv")

# lets print the shape of the dataset
print("The Shape of the Dataset:", data.shape)

# lets check the head of the dataset
data.head()
print (data.head())

# lets Explore Some of the Important Column in the dataset
print("Number of patient comments present in the Dataset:", data['drug_id'].nunique())
print("Number of unique drugs in the Dataset:", data['drug_name'].nunique())
print("Number of Unique Medical Conditions present in the Dataset:", data['condition'].nunique())
print("\nThe Time Period of Collecting the Data")
#print(data['date'])
print("Starting Date:", data[['date']][1:].max())
print("Ending Date:", data[['date']][1:].min())





#Phase2: Summarizing the dataset
#lets summarize the Dataset
print(data[['rating','ADR count']].describe())

#lets check the Number and Name of the Drugs with 0 ADR Count in Details
print("Analysis on Drugs with No ADR")
print("----------------------------")
print("The Number of Drugs with No ADR Count:", data[data['ADR count'] == 0].count()[0])

#lets Check the Number of Drugs with No ADR Count with rating Greater than or Equal to 4
print("Number of patients who were satisfied using each drug(rating>=4) and experienced no ADR:", data[(data['ADR count'] == 0) & (data['rating'] >= 4)].count()[0])

# Lets Check the Average Rating of the Drugs with No ADR Count
print("Average Rating of Drugs with No ADR Count: {0:.2f}".format(data[data['ADR count'] == 0]['rating'].mean()))
print("\nAnalysis on Drugs with most adverse reaction")
print("----------------------------")
print("Maximum ADR count extracted from comment column:", int(data['ADR count'].max()))
print("Average Rating of Drugs with 20+ ADR Counts:", data[data['ADR count'] > 20]['rating'].mean())
print("\nName and Condition of these Drugs: \n\n",data[data['ADR count'] > 20][['drug_id','condition','dosage_duration','rating']].reset_index(drop = True))
# lets summarize Categorical data also
print('\n\n', data[['drug_name','condition','comment','dosage_duration']].describe(include = 'object'))
# as we know that condition is an Important Column, so we will delete all the records where Condition is Missing
data = data.dropna()
# lets check the Missing values now
data.isnull().sum().sum()








#Phase 3. Unveiling Hidden Patterns from the Data:
# lets check the Distribution of Rating and ADR Count
plt.rcParams['figure.figsize'] = (15, 4)
plt.subplot(1, 2, 1)
sns.histplot(data['rating'])
plt.subplot(1, 2, 2)
sns.histplot(data['ADR count'])
plt.suptitle('Distribution of Rating and ADR Count \n ', fontsize = 20)
plt.show()

# lets check the Impact of Ratings on ADR counting
plt.rcParams['figure.figsize'] = (15, 4)
sns.barplot(x=data['rating'], y=data['ADR count'], palette = 'hot')
plt.grid()
plt.xlabel('\n Ratings')
plt.ylabel('Count\n', fontsize = 20)
plt.title('\n Rating vs ADR count \n', fontsize = 20)
plt.show()


# Checking whether Length of comment has any Impact on Ratings of the Drugs
# for that we need to create a new column to calculate length of the comments
data['len']  = data['comment'].apply(len)
# lets check the Impact of Length of comments on Ratings
print(data[['rating','len']].groupby(['rating']).agg(['min','mean','max']))

# lets check the Highest Length comment
print("Length of Longest comment:", data['len'].max())
print(data['comment'][data['len'] == data['len'].max()].iloc[0])









#Phase 4: Cleaning the Reviews
# as it is clear that the comments have so many unnecassry things such as Stopwords, Punctuations, numbers etc
# First lets remove Punctuations from the comments

data['comment'] = data['comment'].apply(punctuation_removal)

# Now lets Remove the Stopwords also
stop = stopwords.words('english')
stop.append("i'm")
stop_words = []

for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item) 

data['comment'] = data['comment'].apply(stopwords_removal)

# lets remove the Numbers also
data['comment'] = data['comment'].apply(drop_numbers)
#print('\n',data['comment'])








#Phase 5: Calculating the Sentiment from Reviews
# for using Sentiment Analyzer we will have to download the Vader Lexicon from NLTK
#nltk.download('vader_lexicon')
# lets calculate the Sentiment from comments

sid = SentimentIntensityAnalyzer()
train_sentiments = []

for i in data['comment']:
    train_sentiments.append(sid.polarity_scores(i).get('compound'))
    
train_sentiments = np.asarray(train_sentiments)
data['sentiment'] = pd.Series(data=train_sentiments)
# lets check Impact of Sentiment on comments
print(data[['rating','sentiment']].groupby(['rating']).agg(['min','mean','max']))
# as we can see that Sentiment and length of the comment are not related to comments, we will drop the sentiment column
# lets remove the date, comment, len, and sentiment column also
data = data.drop(['date','sentiment','comment','len'], axis = 1)
# lets check the name of columns now
print(data.columns)








#phase6: Calculating the Effectiveness and Usefulness of Drugs
# Lets Calculate an Effective Rating
min_rating = data['rating'].min()
max_rating = data['rating'].max()
data['eff_score'] = data['rating'].apply(scale_rating)
# lets also calculate ADR count Score
data['usefulness'] = data['rating']*data['ADR count']*data['eff_score']
#print(data['usefulness'])

# lets check the Top 10 Most Useful Drugs with their Respective Conditions
print(data[['drug_id','condition','usefulness','dosage_duration']][data['usefulness'] > data['usefulness'].mean()].sort_values(by = 'usefulness', ascending = False).head(10).reset_index(drop = True))






#phase7: Analyzing the Medical Conditions:
# lets check the Most Common Conditions
print("Number of Unique Conditions :", data['condition'].nunique(),'\n')
print(data['condition'].value_counts().head(10))
# lets remove all the Duplicates from the Dataset
data = data.drop_duplicates()







#Phase 8: Finding Most Useful and Useless Drugs for each Condition
# lets find the Highest and Lowest Rated Drugs for each Condition
@interact
def high_low_rate(condition = list(data['condition'].value_counts().index)):
    print("\n Top 5 Usefull Results")
    print(data[data['condition'] == condition][['drug_id','usefulness','dosage_duration']].sort_values(by = 'usefulness', ascending = False).head().reset_index(drop = True))

    print("\n\n Bottom 5 Usefull Results")
    print(data[data['condition'] == condition][['drug_id','usefulness','dosage_duration']].sort_values(by = 'usefulness', ascending = True).head().reset_index(drop = True))
