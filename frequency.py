import csv
import nltk
from string import punctuation
from operator import itemgetter
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer


def lemmatize_shit():  # switch to a PG title for delicate eyes
    lemmatizer = WordNetLemmatizer()

    with open('tweets_rated.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)

        with open('lemmas.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            for row in reader:
                outfile.write(row[0])
                outfile.write(',')
                tokens = nltk.word_tokenize(row[1])
                # part of speech
                tagged_tokens = nltk.pos_tag(tokens)
                # print(tagged_tokens)

                # Lemmatize the tokensies
                lemmatized_tokens = []
                for token, tag in tagged_tokens:
                    if tag.startswith('N'):
                        # Noun
                        lemmatized_token = lemmatizer.lemmatize(token, pos='n')
                    elif tag.startswith('V'):
                        # Verb
                        lemmatized_token = lemmatizer.lemmatize(token, pos='v')
                    elif tag.startswith('J'):
                        # Adjective
                        lemmatized_token = lemmatizer.lemmatize(token, pos='a')
                    elif tag.startswith('R'):
                        # Adverb
                        lemmatized_token = lemmatizer.lemmatize(token, pos='r')
                    else:
                        # Other
                        lemmatized_token = lemmatizer.lemmatize(token)
                    lemmatized_tokens.append(lemmatized_token)

                # Write the lemmatized text to the output CSV file
                writer.writerow([' '.join(lemmatized_tokens)])


def make_words_dict():
    words = {}
    words_gen = (word.strip(punctuation).lower() for line in open("lemmas.csv")
                 for word in line.split())

    for word in words_gen:
        words[word] = words.get(word, 0) + 1

    top_words = sorted(words.items(), key=itemgetter(
        1), reverse=True)  # add limit if wanted, eg: [:2000]

    file = open('make_words_dict.txt', 'w')
    for row in top_words:
        print(row[0])
        file.write(row[0]+" ")
    file.close()


lemmatize_shit()
make_words_dict()
