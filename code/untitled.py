import pandas as pd
# import matplotlib.pyplot as plt
from autocorrect import spell

cvss = ['ConanOBrien_tweets.csv','cristiano_tweets.csv',
        'donaldTrump_tweets.csv','ellenShow_tweets.csv',
        'jimmykimmel_tweets.csv','joeBiden_tweets.csv',
        'KimKardashian_tweets.csv','labronJames_tweets.csv',
        'ladygaga_tweets.csv','Schwarzenegger_tweets.csv']

BIG_A_ORD = ord('A')
BIG_Z_ORD = ord('Z')
LITTLE_A_ORD = ord('a')
LITTLE_Z_ORD = ord('z')
MAX_SPECIAL_ORD = 47
COMMA_ORD = 44
EXCLAMATION_ORD = 33
DOT_ORD = 63


longest_sequence_letters = dict()
mean_of_length = dict()
mean_length_of_tweets = dict()
dots = dict()
commas = dict()
exclamation_mark = dict()
retweeted = dict()
misspelling = dict()

maps = [longest_sequence_letters,mean_of_length,mean_length_of_tweets,
        dots,commas,exclamation_mark,retweeted,misspelling]
names_maps = ['longest_sequence_letters','mean_of_length',
              'mean_length_of_tweets','dots','commas','exclamation_mark',
              'retweeted','misspelling']
ignore = ['@','#','http','&']


def longest_capital_sequence (tweet):

    counter = 0
    longest = 0
    ocurrence = 0
    mean = 0
    num_commas = 0
    num_exclamation = 0
    num_dots = 0
    retweeted = 0
    num_misspelling = num_of_missplling(tweet,"")

    for i in range(len(tweet)):
        current_order = ord(tweet[i])
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
    if(tweet[:2] == 'RT'):
        retweeted = 1

    return [longest, mean, len(tweet), num_commas, num_exclamation, num_dots,
            retweeted, num_misspelling]

# def mean_capital_sequence_len(tweet):
#     counter = 0
#     occurance = 0
#     mean = 0
#     num_commas = 0
#     num_exclamation = 0
#     num_dots = 0
#
#
#     for i in range(len(tweet)):
#         current_order = ord(tweet[i])
#         if (not LITTLE_A_ORD<=current_order<=LITTLE_Z_ORD):
#             if (BIG_A_ORD<=current_order<=BIG_Z_ORD):
#                 counter += 1
#
#         else:
#             if(counter != 0):
#                 ocurrence +=1
#                 mean += counter
#                 counter = 0
#     return mean/counter
#
# def len_of_tweet(tweet):
#     return len(tweet)
#
# def number_of_dots




def num_of_missplling(tweet,name):
    words = tweet.split()
    counter = 0
    # words = speller.unknown(words)
    for word in words:
        if(word.isalpha()):
            if(spell(word) != word):
                if (name != ""):
                    misspelling[name] +=1
                counter += 1
    return counter





for cvs in cvss:
    data = pd.read_csv(""+cvs)
    data = data.iloc[:,1:].as_matrix()
    ocurrence = 0
    tweets_len = 0
    name = cvs[:6]

    commas[name] = 0
    exclamation_mark[name] = 0
    dots[name] = 0
    retweeted[name] = 0
    misspelling[name] = 0

    for tweet in data:
        tweet = tweet[0][2:-2]
        num_of_missplling(tweet,name)
print(misspelling)

    #     counter = 0
    #     tweets_len += len(tweet)
    #
    #     if (tweet[:2] == 'RT'):
    #         retweeted[name] += 1
    #
    #     for i in range(len(tweet)):
    #         current_order = ord(tweet[i])
    #         if (not LITTLE_A_ORD<=current_order<=LITTLE_Z_ORD):
    #
    #             if (BIG_A_ORD<=current_order<=BIG_Z_ORD):
    #                 counter += 1
    #             else:
    #                 if (current_order == COMMA_ORD):
    #                     commas[name] += 1
    #                 elif (current_order == EXCLAMATION_ORD):
    #                     print (tweet[i])
    #                     print (name)
    #                     exclamation_mark[name] += 1
    #                     print (exclamation_mark[name])
    #                 elif (current_order == DOT_ORD):
    #                     dots[name] += 1
    #
    #         else:
    #             if(counter !=0):
    #                 if (ocurrence == 0):
    #                     mean_of_length[name] = counter
    #                     longest_sequence_letters[name] = counter
    #                 else:
    #                     mean_of_length[name] += counter
    #                     if (longest_sequence_letters[name]<counter):
    #                         longest_sequence_letters[name] = counter
    #                 counter = 0
    #                 ocurrence +=1
    #
    #     if (counter != 0):
    #         if (ocurrence == 0):
    #             mean_of_length[name] = counter
    #             longest_sequence_letters[name] = counter
    #         else:
    #             mean_of_length[name] += counter
    #             if (longest_sequence_letters[name]<counter):
    #                 longest_sequence_letters[name] = counter
    #
    #         counter = 0
    #         ocurrence +=1
    #
    # print (exclamation_mark[name])
    # if (ocurrence != 0):
    #     mean_of_length[name] /= ocurrence
    # mean_length_of_tweets[name] = tweets_len/len(data)
    # commas[name] /= len(data)
    # exclamation_mark[name] /= len(data)
    # dots[name] /= len(data)

# print("mean len of capitals seq")
# print(mean_of_length)
# print("longest seq")
# print(longest_sequence_letters)
# print("mean len of tweets")
# print(mean_length_of_tweets)
# print("dots mean")
# print(dots)
# print("commas mean")
# print(commas)
# print("exc mean")
# print(exclamation_mark)
#
# for i,map in enumerate(maps):
#     temp = sorted(map.items(), key=lambda kv: kv[1])
#     x = []
#     y = []
#     for seq in temp:
#         x.append(seq[0])
#         y.append(seq[1])
#     plt.bar(x,y,align = 'center')
#     plt.title(names_maps[i])
#     plt.show()

