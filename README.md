# Sentiment Analysis of Tweets using NLTK and Machine Learning

This is a python based R&D project for VI semester of my undergraduate program. The aim is to classify tweets, whether they are positive or negative using multiple ML algorithms and identify if any particular algorithm performs better than others.

## Corpus
NLTK corpus has been used to extract the twitter samples

```python
from nltk.corpus import twitter_samples
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
```
This dataset consists of 5000 tweets in each, positive tweets and negative tweets, resulting a total of 10,000 tweets in total. 

## PreProcessing

We define a function `clean_tweets(tweet)` which cleans a tweet of string type by using regular expression. We clean stock market tickers, old style retweet texts i.e 'RT', remove hyperlinks, remove hashtags and then tokenize the tweets using `TweetTokenizer`.

```python
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
```

We also define two sets of emoticons, a happy set and a sad set.
```python
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
```

Now, we have two methods of cleaning and stemming:

- Remove all stopwords, emoticons and any strings that are not alphabetic words
```python
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in emoticons and  # remove emoticons
            word not in string.punctuation and # remove punctuation
                re.fullmatch(re.compile(r"[A-Za-z]+"), word)):  
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
```

- Remove stopwords and non-words but preserve emoticons
```python
    for word in tweet_tokens:
        if(word in stopwords_english):
            continue
        if(word in emoticons or re.fullmatch(re.compile(r"[A-Za-z]+"), word)):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
```

We then define a `bag_of_words(tweet)` function which makes use of the `clean_tweets(tweet)` function defined above and returns us a bag of words which can be easily be used as a feature set in our training algorithm.

```python
def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary
```

## Feature Set and Splitting

We define the feature set using the `bag_of_words(tweet)` function

```python
# positive tweets feature set
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))

# negative tweets feature set
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))
```

We shuffle the feature sets so at every iteration of training the accuracy differs, then we perform a 80:20 split which gives us the best results in our case.

```python
shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
```

## Training 

We use 3 classifiers for our training, the Naive Bayes Classifier, the Maximum Entropy Classifier and the Support Vector Machine Classifier.

```python
NBclassifier = NaiveBayesClassifier.train(train_set)

MaxEntClassifier = MaxentClassifier.train(
    train_set, 'GIS', trace=0, encoding=None, labels=None,
    gaussian_prior_sigma=0, max_iter=1)

SVCclassifier = SklearnClassifier(LinearSVC(), sparse=False)
SVCclassifier.train(train_set)
```

## Testing And Accuracy

We perform testing of the trained classifier and find out the accuracy.

```python
NBaccuracy = classify.accuracy(NBclassifier, test_set)
print("Naive Bayes Accuracy: " + str(NBaccuracy))

MaxEntAccuracy = classify.accuracy(MaxEntClassifier, test_set)
print("Maximum Entropy Accuracy: " + str(MaxEntAccuracy))

SVCaccuracy = classify.accuracy(SVCclassifier, test_set)
print("SVC Accuracy: "+str(SVCaccuracy))
```

Our accuracy greatly varies depending the type of cleaning that we have used.
- If we remove the emoticons and consider just the words, then we get an accuracy in the range of **75%-80%** where all 3 classifiers perform equally with a negligible variation in accuracy.
- If we do consider the emoticons during the training phase, then the accuracy greatly increases by 10-15%, resulting in a accuracy of **90%-95%**.

## Custom Rating

While classification into tweets into positive and negative was the goal, but it would even more productive if we could not only classify but also define the magnitude of their positivity. 

Clearly the sentences, `'The food was good.'` and `'The food was absolutely delicious and the service was amazing'`, are very different in magnitudes though they both would be classified into 'positive'.

In order to implement a simple rating system, we would need a corpus of words which defines the magnitude of positivity/negativity of each word. Thus, we define the following:

```python
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
```

We then define a `custom_rating` function which considers the weight of all positive and negative words in the tweet using the corpus defined above and calculates the rating percentage.

```python
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
```

## Usage

In order to use the project, the following dependencies are required:
- NLTK and NLTK data, can be installed by following the steps [here](http://www.nltk.org/install.html)
- Scikit-learn, can be installed by following the steps [here](https://scikit-learn.org/stable/install.html)

Run the source code in a jupyter-like notebook environment and call the function `classify_for_custom_input(custom_tweet: str)` where the input parameter is of type `string`. The console would print a rating percentage and the classification result from all 3 classifiers.

```python
def classify_for_custom_input(custom_tweet: str):
    custom_tweet_set = bag_of_words(custom_tweet)

    print("Rating: ", custom_rating(custom_tweet_set=custom_tweet_set), "%")

    result_Naive_Bayes = NBclassifier.classify(custom_tweet_set)
    print("Result from Naive Bayes Classifier: ", result_Naive_Bayes)

    result_MaxEnt = MaxEntClassifier.classify(custom_tweet_set)
    print("Result from MaxEnt Classifier: ", result_MaxEnt)

    result_SVC = SVCclassifier.classify(custom_tweet_set)
    print("Result from SVC Classifier: ", result_SVC)
```

## Drawbacks

**Sarcasm** is a great potential drawback of the current designed system. Sentences like `'Got mugged today. What an amazing day :)'` would throw off the classification and return a undesired result i.e in this case `positive`.