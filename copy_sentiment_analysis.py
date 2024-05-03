import tensorflow as tf
import tf.keras as keras
import pandas as pd
import numpy as np

train_data=pd.read_csv("/english_tweets-train.csv")
test_data=pd.read_csv("/english_tweets-test.csv")
frequency=pd.read_csv("/frequency.csv")

index_file=pd.read_csv("/words.csv")
word_index=dict(zip(index_file.Words, index_file.Indexes))
word_index["excluded"]=0
word_index["included"]=1

def tweet_encoder(tweets):
    arr=[word_index[word] for word in tweets]
    return arr

train_data,train_labels=tweets['Tweets'], tweets['Sentiment']
test_data,test_labels=test_tweets['Tweets'], test_tweets['Sentiment']

# encode them all to 0s/1s
def encode_sentiments(sentiment):
    if sentiment=='positive':
        return 1
    else:
        return 0
train_labels=train_labels.apply(encode_sentiments)
test_labels=test_labels.apply(encode_sentiments)

train_data=train_data.apply(lambda tweet:tweet.split())
test_data=test_data.apply(lambda tweet:tweet.split())

train_data=train_data.apply(tweet_encoder)
test_data=test_data.apply(tweet_encoder)

train_data=keras.preprocessing.sequence.pad_sequence(train_data)


def feature_extractor(data):
    document_words = set(data)
    features = {}
    for word in frequency:
        features['contains({})'.format(word)] = (word in document_words)
    print(features)
    return features
