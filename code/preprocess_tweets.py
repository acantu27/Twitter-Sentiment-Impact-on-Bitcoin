import config
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


class ProcessLanguage:
    """Natural Language Processing (NLP)
    Imports csv file into pandas dataframe (df)
    Sets and drops df columns with configuration information
    Preprocesses data from tweets column in df"""

    def __init__(self, csv_file):
        """Init: initialize csv_file name
        :param csv_file: name of file to be imported"""
        self.csv_file = csv_file

    def import_dataset(self):
        """Import Dataset: Imports csv file into pandas dataframe (df) with no header and 'utf-8' encoding"""
        self.dataframe = pd.read_csv(self.csv_file, encoding='utf-8', header=None)
        return self.dataframe

    def set_columns(self, columns):
        """Set Columns: Set dataframe (df) columns as 'columns' from config.py"""
        self.dataframe.columns = columns

    def drop_columns(self, columns):
        """Drop Columns: Drops columns from df as 'columns' from config.py"""
        self.dataframe.drop(columns, axis=1, inplace=True)

    def clean_tweet(self):
        """Clean Tweet: Implements other functions in the proper order to clean tweets from the pandas df"""
        self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.contraction_expander)
        self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.preprocess_dataframe)
        self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.tokenize_tweet)
        # Removing the stopword func increased testing acc 2%
        #self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.remove_stopwords)
        #self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.revised_stopwords)
        #self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.stemmer)
        #self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.lemmatizer)
        self.dataframe['tweet'] = self.dataframe['tweet'].apply(self.join_words)

    def preprocess_dataframe(self, tweet):
        """Preprocess_dataframe: Basic data removal from tweets
        Lowercase, URL removal, @username removal, HTML decoding, RT, non-alphabetical char, and extra whitespaces"""

        # Convert tweet to all lower case characters
        # Already used in contraction_expander
        # tweet = tweet.lower()

        # Remove all URL's from tweets
        tweet = re.sub(r'http\S+', '', tweet)

        # Removes the @ and username (@bobby -> "")
        tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)

        # HTML Decoding
        # Removes encoded text (&amp; -> &, &quot; -> ")
        # Prevents need to decode with BeautifulSoup
        tweet = re.sub('&amp;', ' and ', tweet)
        tweet = re.sub('&quot;', '"', tweet)

        # Remove "RT"/"rt" from the tweet field ("RT hey there" -> "hey there"
        # API uses this to indicate ReTweet
        tweet = re.sub(',rt', ' ', tweet)
        tweet = re.sub(' rt ', ' ', tweet)

        # Removes all non-alphabetical characters
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)

        # Remove all additional white spaces "  ", "\n", etc
        # Last this be the last option to clear up any white spaces left by the other functions
        tweet = re.sub('[\s]+', ' ', tweet)

        return tweet

    def tokenize_tweet(self, tweet):
        """Tokenize Tweets: Break sentences down to individual words for processing."""
        tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
        return tknzr.tokenize(tweet)

    # ##################################################################################
    # https: // en.wikipedia.org / wiki / English_auxiliaries_and_contractions
    # ##################################################################################
    def contraction_expander(self, tweet):
        """Contraction Expander: Expands all common contractions using dict.
        Examples: (don't -> do not), (y'all -> you all), (who'll -> who will)"""

        # Must be converted to lowercase or else add additional entries into dict with capitals included
        tweet = tweet.lower()

        expand = re.compile('(%s)' % '|'.join(config.contraction_exp.keys()))

        def replace(match):
            return config.contraction_exp[match.group(0)]

        tweet = expand.sub(replace, tweet)

        return tweet

    # ##################################################################################
    # Must be tokenized before using this method
    # Does NOT remove all contractions. Use contraction_expander()
    # to remove all useless data
    # ##################################################################################
    def remove_stopwords(self, tweet):
        """Remove Stopwords: Removes stopwords from tweet df.
        Example: (a, an, the, in)"""
        stop_words = set(stopwords.words('english'))

        filter_tweet = []

        for w in tweet:
            if w not in stop_words:
                filter_tweet.append(w)

        tweet = filter_tweet

        return tweet

    def revised_stopwords(self, tweet):
        """Revised Stopwords: Removes stopwords using a modified list of NLTK Corpus."""
        revised_tweet = []

        for word in tweet:
            if word not in config.revised_stopwords:
                revised_tweet.append(word)

        tweet = revised_tweet

        return tweet

    def join_words(self, tweet):
        """Join Word: Rejoins words after tokenization to reform sentences.
        Example: (['He'], ['is'], ['going'], ['now'])->("He is going now")"""
        tweet = (" ".join(tweet)).strip()
        return tweet

    def stemmer(self, tweet):
        """Stemmer: """
        ps = PorterStemmer()

        stemmed_tweet = []

        for word in tweet:
            stemmed_tweet.append(ps.stem(word))

        tweet = stemmed_tweet

        return tweet

    def lemmatizer(self, tweet):
        """Lemmitizer: """
        ls = WordNetLemmatizer()

        lemmatized_tweet = []

        for word in tweet:
            lemmatized_tweet.append(ls.lemmatize(word))

        tweet = lemmatized_tweet

        return tweet

    def negation_handling(self, tweet):
        """Negation Handling: Converts negative contractions to 'not'.
        Example: (can't -> not), (won't -> not)"""
        return tweet
