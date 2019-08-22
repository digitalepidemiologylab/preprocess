import os
import re
import time
import logging
from utils.helpers import get_merged_data
from nltk import download
from nltk.tokenize import TweetTokenizer
from nltk.corpus import words
from nltk.corpus import stopwords
from collections import Counter

class CleanTweets():
    """Filter operation to get from raw data to cleaned data"""

    def __init__(self, df, verbose=True):
        self.verbose = verbose
        self.df = df
        self.en_stopwords = frozenset([s.lower() for s in stopwords.words('english')])
        self.en_words = frozenset([s.lower() for s in words.words('en')])
        self.logger = logging.getLogger(__name__)
        download('stopwords', quiet=True)
        download('words', quiet=True)

    def remove_duplicates(self):
        """Removes tweets with same ID"""
        self.df.drop_duplicates(subset='id', keep='first', inplace=True)

    def flag_non_unique(self):
        unique_texts = set()
        is_duplicate = []
        for i, t in enumerate(self.df['text']):
            if self.df.iloc[i]['is_retweet']:
                is_duplicate.append(False)
                continue
            if t not in unique_texts:
                unique_texts.add(t)
                is_duplicate.append(False)
            else:
                # susequent appearences of the same text will be marked as non-unique
                is_duplicate.append(True)
        self.df['is_duplicate'] = is_duplicate

    def add_token_count(self):
        tt = TweetTokenizer()
        text = self.df['text'].apply(str)
        text = text.apply(lambda t: re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>', t))
        text = text.apply(lambda t: re.sub('(\@[^\s]+)','@<user>', t))
        text = text.apply(tt.tokenize)
        text = text.apply(lambda tokens: [t.lower() for t in tokens])
        text = text.apply(lambda tokens: [w for w in list(tokens) if str(w) not in self.en_stopwords and str(w) in self.en_words])
        self.df['token_count'] = text.apply(lambda t: len(t))

    def add_decision_flags(self, token_count_cutoff=3):
        """Add flags which have been pre-defined for later processing steps"""
        # labelling: no retweets, no extracted tweets, no duplicates, min token count
        self.df['use_for_labelling'] = (~self.df['is_retweet']) & (~self.df['extracted_quoted_tweet']) & (~self.df['is_duplicate']) & (self.df['token_count'] >= token_count_cutoff)
        # prediction: min token count
        self.df['use_for_prediction'] = self.df['token_count'] >= token_count_cutoff

    def write(self, dtype):
        f_name = 'cleaned_{}.csv'.format(dtype)
        path = os.path.join('data', '2_cleaned', f_name)
        self.logger.info('Writing file {}...'.format(path))
        self.df.to_csv(path, encoding='utf8')

def run(dtypes=['original'], verbose=False):
    s_time = time.time()
    logger = logging.getLogger(__name__)
    for dtype in dtypes:
        logger.info('Reading data of type {}...'.format(dtype))
        df = get_merged_data(dtype=dtype)
        num_tweets_with_duplicates = len(df)
        clt = CleanTweets(df, verbose=verbose)
        clt.remove_duplicates()
        num_tweets = len(df)
        if verbose:
            num_duplicates = num_tweets_with_duplicates - num_tweets
            logger.info('Removed {:,} duplicates from {:,} resulting in {:,} tweets'.format(num_duplicates, num_tweets_with_duplicates, num_tweets))
        clt.flag_non_unique()
        if verbose:
            non_unique_counts = dict(Counter(clt.df.is_duplicate))[True]
            logger.info('Number of text duplicates: {:,}/{:,} ({:.1f}%)'.format(non_unique_counts, num_tweets, 100*non_unique_counts/num_tweets))
        clt.add_token_count()
        if verbose:
            mean_token_count = clt.df.token_count.mean()
            median_token_count = clt.df.token_count.median()
            logger.info('Token counts: Mean: {:.2f}, Median: {:.2f}'.format(mean_token_count, median_token_count))
        clt.add_decision_flags(token_count_cutoff=3)
        if verbose:
            labelling_counts = dict(Counter(clt.df.use_for_labelling))[True]
            prediction_counts = dict(Counter(clt.df.use_for_prediction))[True]
            logger.info('Marked to be used for annotation: {:,}/{:,} ({:.1f}%)'.format(labelling_counts, num_tweets, 100*labelling_counts/num_tweets))
            logger.info('Marked to be used for prediction: {:,}/{:,} ({:.1f}%)'.format(prediction_counts, num_tweets, 100*prediction_counts/num_tweets))
        clt.write(dtype)
    e_time = time.time()
    logger.info('... done after {:.1f} min'.format((e_time - s_time)/60.0))
