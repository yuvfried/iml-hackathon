import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emoji

files = ['donaldTrump_tweets.csv', 'joeBiden_tweets.csv',
         'ConanOBrien_tweets.csv', 'ellenShow_tweets.csv',
         'KimKardashian_tweets.csv', 'labronJames_tweets.csv',
         'ladygaga_tweets.csv', 'cristiano_tweets.csv',
         'jimmykimmel_tweets.csv', 'Schwarzenegger_tweets.csv']

names = ['Trump', 'Biden', 'Conan', 'Ellen',
         'Kim', 'Lebron', 'Lady Gaga', 'Ronaldo',
         'kimmel', 'schwarzenegger']


def count_emojis(text):
    """
    counts the number of emojis in the text
    :param text: The text to search in
    :return: number of emojis in text
    """
    counter = 0
    for c in text:
        if any(char in emoji.UNICODE_EMOJI for char in c):
            counter += 1

    return counter


avg_emojis = {}
for j in range(len(files)):
    # read only 70% of data
    data = pd.read_csv(files[j])
    data = data[(data.index >= np.percentile(data.index, 0)) & (
        data.index <= np.percentile(data.index, 70))]
    npdata = data.values

    sum_emojis = 0
    # iterate over all tweets and count the number of emojis
    for i in range(npdata.shape[0]):
        tweet = npdata[i, 1]
        sum_emojis += count_emojis(tweet)

    avg_emojis[names[j]] = sum_emojis / npdata.shape[0]

# sort the results by emoji number
sorted_avg_emojis = sorted(avg_emojis.items(), key=lambda kv: kv[1])
x_axis = [0] * 10
y_axis = [0] * 10
for i in range(len(x_axis)):
    x_axis[i] = sorted_avg_emojis[i][0]
    y_axis[i] = sorted_avg_emojis[i][1]


# plot the graph
plt.bar(range(len(x_axis)), y_axis, align='center')
plt.xticks(range(len(x_axis)), x_axis)
plt.title("Average emojis per tweet")
plt.xlabel('celebrity')
plt.ylabel('emoji number')
plt.show()
