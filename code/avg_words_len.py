import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

files = ['donaldTrump_tweets.csv', 'joeBiden_tweets.csv',
         'ConanOBrien_tweets.csv', 'ellenShow_tweets.csv',
         'KimKardashian_tweets.csv', 'labronJames_tweets.csv',
         'ladygaga_tweets.csv', 'cristiano_tweets.csv',
         'jimmykimmel_tweets.csv', 'Schwarzenegger_tweets.csv']

names = ['Trump', 'Biden', 'Conan', 'Ellen',
         'Kim', 'Lebron', 'Lady Gaga', 'Ronaldo',
         'kimmel', 'schwarzenegger']


def get_avg_word_len(text):
    """
    returns the average length of word in the text
    :param text: the text to search in
    :return: the average length of word in the text
    """
    # remove links, hashtags, @ and punctuation marks
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[,.)(!?:\[\]\"\']', '', text)

    # split to words
    words = re.findall(r'\S+', text)
    if len(words) == 0:
        return 0
    words_lens = [len(word) for word in words]

    # calc len of longest word
    return sum(words_lens) / len(words)


avg_longest = {}
for j in range(len(files)):
    # read only 70% of data
    data = pd.read_csv(files[j])
    data = data[(data.index >= np.percentile(data.index, 0)) & (
        data.index <= np.percentile(data.index, 70))]
    npdata = data.values

    sum_avg_word_len = 0
    # iterate the tweets and get average word length
    for i in range(npdata.shape[0]):
        tweet = npdata[i, 1]
        sum_avg_word_len += get_avg_word_len(tweet)

    avg_longest[names[j]] = sum_avg_word_len / npdata.shape[0]

print(avg_longest)
sorted_avg_word_len = sorted(avg_longest.items(), key=lambda kv: kv[1])
print(sorted_avg_word_len)
x_axis = [0] * 10
y_axis = [0] * 10
for i in range(len(x_axis)):
    x_axis[i] = sorted_avg_word_len[i][0]
    y_axis[i] = sorted_avg_word_len[i][1]


plt.bar(range(len(x_axis)), y_axis, align='center')
plt.xticks(range(len(x_axis)), x_axis)
plt.title("Average words length by celebrity")
plt.xlabel('celebrity')
plt.ylabel('word length')
plt.show()
