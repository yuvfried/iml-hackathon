import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from autocorrect import spell
import re
import emoji

zero = pd.read_csv("donaldTrump_tweets.csv")
one = pd.read_csv("joeBiden_tweets.csv")
two = pd.read_csv("ConanOBrien_tweets.csv")
three = pd.read_csv("ellenShow_tweets.csv")
four = pd.read_csv("KimKardashian_tweets.csv")
five = pd.read_csv("labronJames_tweets.csv")
six = pd.read_csv("ladygaga_tweets.csv")
seven = pd.read_csv("cristiano_tweets.csv")
eight = pd.read_csv("jimmykimmel_tweets.csv")
nine = pd.read_csv("Schwarzenegger_tweets.csv")

# ROWS = 1200
#
# zero = pd.read_csv("donaldTrump_tweets.csv", nrows=ROWS)
# one = pd.read_csv("joeBiden_tweets.csv", nrows=ROWS)
# two = pd.read_csv("ConanOBrien_tweets.csv", nrows=ROWS)
# three = pd.read_csv("ellenShow_tweets.csv", nrows=ROWS)
# four = pd.read_csv("KimKardashian_tweets.csv", nrows=ROWS)
# five = pd.read_csv("labronJames_tweets.csv", nrows=ROWS)
# six = pd.read_csv("ladygaga_tweets.csv", nrows=ROWS)
# seven = pd.read_csv("cristiano_tweets.csv", nrows=ROWS)
# eight = pd.read_csv("jimmykimmel_tweets.csv", nrows=ROWS)
# nine = pd.read_csv("Schwarzenegger_tweets.csv", nrows=ROWS)
dfs = [zero, one,two,three,four,five,six,seven,eight,nine]
# concating to 1 df
df = zero
for data in dfs[1:]:
    df = df.append(data,ignore_index= True)

# magic numbers
BIG_A_ORD = ord('A')
BIG_Z_ORD = ord('Z')
LITTLE_A_ORD = ord('a')
LITTLE_Z_ORD = ord('z')
MAX_SPECIAL_ORD = 47
COMMA_ORD = 44
EXCLAMATION_ORD = 33
DOT_ORD = 63

# splitting to train-test
train = df.sample(frac=0.8)
test = df.drop(train.index)

def num_of_missplling(tweet):
    words = tweet.split()
    counter = 0
    # words = speller.unknown(words)
    for word in words:
        if(word.isalpha()):
            if(spell(word) != word):
                counter += 1
    return counter


# function which produce some 'aprioric' features
def get_manual_features(tweet):
    temp_tweet = tweet[2:-2]
    counter = 0
    longest = 0
    ocurrence = 0
    mean = 0
    num_commas = 0
    num_exclamation = 0
    num_dots = 0
    retweeted = 0
    max_word_len = 0
    avg_word_len = 0
    emoji_count = 0
    spell_count = 0

    for c in temp_tweet:
        if any(char in emoji.UNICODE_EMOJI for char in c):
            emoji_count += 1

    #count the misspels
    words = tweet.split()
    for word in words:
        if(word.isalpha()):
            if(spell(word) != word):
                spell_count += 1

    # filter the tweet
    temp_tweet_2 = re.sub(r'https?://\S+', '', temp_tweet)
    temp_tweet_2 = re.sub(r'#\S+', '', temp_tweet_2)
    temp_tweet_2 = re.sub(r'@\S+', '', temp_tweet_2)
    temp_tweet_2 = re.sub(r'[,.)(!?:\[\]\"\']', '', temp_tweet_2)
    # split to words
    words = re.findall(r'\S+', temp_tweet_2)

    if len(words) != 0:
        # calc len of longest word
        max_word_len = len(max(words, key=len))

        # calc avg len of words
        words_lens = [len(word) for word in words]
        avg_word_len = sum(words_lens) / len(words)

    for i in range(len(temp_tweet)):
        current_order = ord(temp_tweet[i])
        if (not LITTLE_A_ORD<=current_order<=LITTLE_Z_ORD):
            if (BIG_A_ORD<=current_order<=BIG_Z_ORD):
                counter += 1
            else:
                if (current_order == COMMA_ORD):
                    num_commas += 1
                elif (current_order == EXCLAMATION_ORD):
                    num_exclamation += 1
                elif (current_order == DOT_ORD):
                    num_dots += 1
        else:
            if(counter !=0):
                if(counter>longest):
                    longest = counter
                ocurrence +=1
                mean += counter
                counter = 0
    if(ocurrence != 0):
        mean = mean/ocurrence
    if(temp_tweet[:2] == 'RT'):
        retweeted = 1

    return [longest, mean, len(tweet), num_commas, num_exclamation, num_dots, retweeted, max_word_len, avg_word_len, emoji_count,spell_count]

# apply the above to the whole df
def apply_manuals(tweets_array):
    s_of_lists = tweets_array.apply(lambda t: get_manual_features(t))
    manuals_df = s_of_lists.apply(lambda x:pd.Series(x))
    manuals_df.columns = ['longest', 'mean', "len_of_tweet", 'num_commas', 'num_exclamation', 'num_dots','retweeted', 'max_word_len', 'avg_word_len', 'emoji_count', 'spell_count']
    return manuals_df

# function to stem one tweet
def stem_tweet(tweet):
    new_tweet=[]
    ps = PorterStemmer()
    for w in tweet:
        new_tweet.append(ps.stem(w))
    return new_tweet

def pre_processing_train(tweets_array):
    manuals_features = apply_manuals(tweets_array).values
    tweets_array = tweets_array.str.split().apply(lambda t: stem_tweet(t))
    tweets_array = tweets_array.apply(lambda lst: ' '.join(lst))
    vec = CountVectorizer()
    bow = vec.fit_transform(tweets_array)
    vocab = vec.vocabulary_
    train_X = bow.toarray()
    print(train_X.shape)
    train_X = np.concatenate((train_X, manuals_features), axis=1)
    return train_X, vocab

def pre_processing_test(tweets_array, vocab):
    print("start pre-process test")
    manuals_features = apply_manuals(tweets_array).values
    tweets_array = tweets_array.str.split().apply(lambda t: stem_tweet(t))
    tweets_array = tweets_array.apply(lambda lst: ' '.join(lst))
    vec = CountVectorizer(vocabulary=vocab)
    bow = vec.transform(tweets_array)
    test_X = bow.toarray()
    test_X = np.concatenate((test_X, manuals_features), axis=1)
    print("end pre-process test")
    return test_X

print('functions defined')

train_X, vocab = pre_processing_train(train['tweet'])
train_y = train.user.values
print('train created')
test_X, test_y = pre_processing_test(test.tweet, vocab), test.user.values
print('test created')

def apply_logreg(train_X,train_y,test_X,test_y):
    print("lambda = 0.5")
    logreg = LogisticRegression(penalty='l1', C=2, solver='liblinear') # lasso with lambda=0.5
    print('fitting...')
    logreg.fit(train_X, train_y)
    print('done fitting')
    print("coef_shape: ", logreg.coef_.shape, "num of zero coefs: ", abs(np.sum(logreg.coef_ < 0.0000001))/10)
    train_pred = logreg.predict(train_X)
    test_pred = logreg.predict(test_X)
    train_error = 1- np.sum(train_pred == train_y)/train_y.shape[0]
    test_error = 1- np.sum(test_pred == test_y)/test_y.shape[0]
    return test_pred, train_error, test_error
#
# def apply_knn(train_X,train_y,test_X,test_y, k):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     neigh.fit(train_X, train_y)
#     pred = neigh.predict(test_X)
#     train_error = 1 - np.sum(pred == train_y)/train_y.shape[0]
#     test_error = 1 - np.sum(pred == test_y)/test_y.shape[0]
#     return pred, train_error, test_error

pred, train_error, test_error = apply_logreg(train_X,train_y,test_X,test_y)
print("train error: ",train_error)
print("test error: ",test_error)
