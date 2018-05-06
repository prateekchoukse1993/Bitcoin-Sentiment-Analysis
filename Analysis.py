
# coding: utf-8

# In[29]:


#Import the necessary methods
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
import time
import datetime
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')


#Functions

def modify_timestamp(timestamp):
    return time.strftime('%Y-%m-%d %H', time.strptime(timestamp,'%a %b %d %H:%M:%S +0000 %Y'))

def format_timestamp(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(timestamp,'%a %b %d %H:%M:%S +0000 %Y'))
	
def unix_to_timestamp(unix_timestamp):
    return datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%Y-%m-%d %H')

def timestamp_to_year(timestamp):
    return str(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H").timetuple()[0])

def remove_hashtags_urls(text):
    text = re.sub(r'(@\S+)', "", text)
    text = re.sub(r'(http\S+)', "", text)
    return re.sub(r'#',"", text)

def final_sentiment_score(row):
    if (row['Compound_Score'] <= -(0.5)):
        val = -1
    elif (row['Compound_Score'] >= (0.5)):
        val = 1
    else:
        val = 0
    return val


# In[71]:


#Reading data from 'data.txt' after extraction

current_directory = os.getcwd()
tweet_analysis = pd.DataFrame()
folder = "\\VM_final data\\"
if "VM_final data" in os.listdir('.'):
    os.chdir(current_directory+folder)
    for name in os.listdir('.'):
        print(name+" processing...")
        tweets_file = open(name, "r")
        created_at = []
        text = []
        unique_id = []
        tweets_file = open(name, "r")
        flag1 = 0
        flag2 = 0
        data = []

        for line in tweets_file:
            flag1 = flag1 + 1
            try:
                tweet = json.loads(line)
                for key, value in tweet.items():
                    if((key is not None) and (value is not None)):
                        #print(key)
                        if(key in 'id'):
                            unique_id.append(value)
                        elif(key in 'text'):
                            text.append(value)
                        elif(key in 'created_at'):
                            created_at.append(value)
            except:
                continue

        tweets = pd.DataFrame(
            {'unique_id': unique_id,
             'created_at': created_at,
             'text': text
            })
        
        
        #Data Cleaning and Manipulation
        
        tweets['created_at'] = tweets['created_at'].apply(lambda x: modify_timestamp(x))
        tweets['text'] = tweets['text'].apply(lambda x: remove_hashtags_urls(x))
        
		#Sentiment Analysis
        sid = SentimentIntensityAnalyzer()
        Sentiment_score = []

        for text in tweets['text']:
            results = sid.polarity_scores(str(text))
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]

            Sentiment_score.append({"Compound_Score": compound,
                          "Positive_Score": pos,
                          "Neutral_Score": neu,
                          "Negative_Score": neg})

        tweets = pd.concat([tweets,pd.DataFrame(Sentiment_score)], axis=1)
        tweets['final_score'] = tweets.apply(final_sentiment_score, axis=1)
        tweets['final_score'] = tweets['final_score'].apply(str)
        tweets = (tweets.groupby(['created_at','final_score']).count().reset_index())[['created_at','final_score','text']]
        tweet_analysis = tweet_analysis.append(tweets, ignore_index=True)
else:
    print("Directory not present")

tweet_analysis.to_csv('..\\tweet_analysis.csv',index = False)
os.chdir("..\\")


# In[75]:


#Manipulating btc data
btc_data = (pd.read_csv('.coinbaseUSD.csv', names = ["unix_timestamp", "price", "amount"]))
#Averaging prices, btc amount for multiple transactions at same unix timestamp
btc_data = btc_data.groupby(['unix_timestamp'], as_index=False)['price','amount'].mean()
btc_data['timestamp'] = btc_data['unix_timestamp'].apply(lambda x: unix_to_timestamp(x))

btc_hourly_data = btc_data.groupby(['timestamp'], as_index=False)['price','amount'].mean()
btc_hourly_data['year'] = btc_hourly_data['timestamp'].apply(lambda x: timestamp_to_year(x))
btc_hourly_data = btc_hourly_data.loc[btc_hourly_data['year']=='2018']
btc_hourly_data = btc_hourly_data.drop(['year', 'amount'], axis=1)
btc_hourly_data.to_csv('btc_data.csv',index = False)
btc_hourly_data['timestamp'] = pd.to_datetime(btc_hourly_data['timestamp'].str.replace(".", ":"))


# In[77]:


#Input for SAS Minor - Input for Time series analysis
positive_tweets = []
negative_tweets = []
neutral_tweets = []
timestamp = []
tweet_analysis = pd.read_csv('tweet_analysis.csv')
count=0
iter_timestamps = tweet_analysis.created_at.unique()
for x in iter_timestamps:
    #print('inside hours')
    for index, row in tweet_analysis.iterrows():
        #print(count)
        if(row[0] == x):
            #print('inside score evaluation')
            if(int(row[1]) == 1):
                positive_tweets.append(row[2])
            elif(int(row[1]) == -1):
                negative_tweets.append(row[2])
            else:
                neutral_tweets.append(row[2])
            if(count==2):
                timestamp.append(row[0])
                #date.append(row[0])
                #print(count)
                count=0
                #print('breaking..')
                break
            else:
                count=count+1        
        else:
            continue

tweet_analysis = pd.DataFrame(
    {'timestamp': timestamp,
     'positive_tweets': positive_tweets,
     'negative_tweets': negative_tweets,
     'neutral_tweets': neutral_tweets
    })

tweet_analysis['timestamp'] = pd.to_datetime(tweet_analysis['timestamp'].str.replace(".", ":"))
tweet_analysis.to_csv('tweet_analysis.csv')

# In[103]:
mapped = pd.DataFrame()
mapped = pd.merge(tweet_analysis,btc_hourly_data, how='left')
mapped.to_csv('mapped.csv',index = False)

