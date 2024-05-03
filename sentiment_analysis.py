import random
import pandas as pd
import numpy as np
import nltk

frequency = open("frequency.txt").read()  # word dict
frequency_list = frequency.split(' ')

tweets = pd.read_csv("lemmas.csv")  # already cleaned/ lemmatized tweets
tweets = tweets.values.tolist()


def feature_extractor(document_words):
    #document_words = set(doc)
    #print("doc!: ", document_words)
    features = {}
    for word in frequency_list:
        features[(word)] = (word in document_words)
        # print("features:", word, features[(word)])
    return features


def main():
    featuresets = [(feature_extractor(d), c) for (c, d) in tweets]
    # seperate train/test
    train_set, test_set = featuresets[75:], featuresets[:75]
    # do the bayyyyes.... look more into different classifiers? hmmmhmhm
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # test results --print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(50)

    # add quick io input to change file namees


if __name__ == "__main__":
    main()
