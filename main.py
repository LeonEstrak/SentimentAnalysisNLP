from sklearn.svm import SVC, LinearSVC
from random import shuffle
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.classify import MaxentClassifier
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
import string
from nltk import stem
from nltk.corpus import twitter_samples

pos_tweets = twitter_samples.strings('positive_tweets.json')
# print (len(pos_tweets)) # Output: 5000

neg_tweets = twitter_samples.strings('negative_tweets.json')
# print (len(neg_tweets)) # Output: 5000

all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
# print (len(all_tweets)) # Output: 20000

#---------------------------------------------------------------------#

stopwords_english = stopwords.words('english')

stemmer = PorterStemmer()


# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)


def clean_tweets(tweet) -> list:
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []

    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in emoticons and  # remove emoticons
            word not in string.punctuation and  # remove punctuation
                re.fullmatch(re.compile(r"[A-Za-z]+"), word)): 
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    # for word in tweet_tokens:
    #     if(word in stopwords_english):
    #         continue
    #     if(word in emoticons or re.fullmatch(re.compile(r"[A-Za-z]+"), word)):
    #         stem_word = stemmer.stem(word)  # stemming word
    #         tweets_clean.append(stem_word)

    return tweets_clean


#---------------------------------------------------------------------#

# feature extractor function


def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


custom_tweet = "RT @Twitter Hello There! Have a great day. :) #good #morning"

'''
print (bag_of_words(custom_tweet))

Output:
 
{'great': True, 'good': True, 'morning': True, 'hello': True, 'day': True}
'''

# positive tweets feature set
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))

# negative tweets feature set
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))

shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]

#---------------------------------------------------------------------#


NBclassifier = NaiveBayesClassifier.train(train_set)

MaxEntClassifier = MaxentClassifier.train(
    train_set, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter=1)

SVCclassifier = SklearnClassifier(LinearSVC(), sparse=False)
SVCclassifier.train(train_set)

NBaccuracy = classify.accuracy(NBclassifier, test_set)
print("Naive Bayes Accuracy: " + str(NBaccuracy))
MaxEntAccuracy = classify.accuracy(MaxEntClassifier, test_set)
print("Maximum Entropy Accuracy: " + str(MaxEntAccuracy))
SVCaccuracy = classify.accuracy(SVCclassifier, test_set)
print("SVC Accuracy: "+str(SVCaccuracy))

#---------------------------------------------------------------------#

good_words = {
    "perfect": 1.0,
    "excellent": 0.9,
    "outstanding": 0.9,
    "beautiful": 0.9,
    "fabulous": 0.9,
    "fantastic": 0.9,
    "amazing": 0.8,
    "incredible": 0.8,
    "delightful": 0.8,
    "spectacular": 0.8,
    "lovely": 0.7,
    "delicious": 0.7,
    "unique": 0.6,
    "good": 0.6,
    "okay": 0.4,
    "acceptable": 0.2,
    "bearable": 0.1,
}

bad_words = {
    "disgusting": -1.0,
    "tasteless": -0.9,
    "unaccpetable": -0.9,
    "narrowminded": -0.9,
    "unbearable": -0.9,
    "inedible": -0.8,
    "abyssmal": -0.8,
    "boring": -0.8,
    "mundane": -0.7,
}


def stem_words_from_dict(dict1: dict, dict2: dict) -> dict:
    result = dict()

    for word in dict1:
        result.update({stemmer.stem(word): dict1.get(word)})
    for word in dict2:
        result.update({stemmer.stem(word): dict2.get(word)})

    return result


def custom_rating(custom_tweet_set: list):
    all_words = stem_words_from_dict(good_words, bad_words)
    word_weight = 0
    word_count = 0
    for word in custom_tweet_set:
        if(word in all_words):
            word_weight += all_words.get(word)
            word_count += 1

    if(word_count != 0):
        return word_weight*100/word_count
    return 0


def classify_for_custom_input(custom_tweet: str):
    custom_tweet_set = bag_of_words(custom_tweet)

    print("Rating: ", custom_rating(custom_tweet_set=custom_tweet_set), "%")

    result_Naive_Bayes = NBclassifier.classify(custom_tweet_set)
    print("Result from Naive Bayes Classifier: ", result_Naive_Bayes)

    result_MaxEnt = MaxEntClassifier.classify(custom_tweet_set)
    print("Result from MaxEnt Classifier: ", result_MaxEnt)

    result_SVC = SVCclassifier.classify(custom_tweet_set)
    print("Result from SVC Classifier: ", result_SVC)
