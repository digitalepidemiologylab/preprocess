import os
import re
import time
import logging
from utils.helpers import get_merged_data
from collections import Counter
import spacy

class CleanTweets():
    """Filter operation to get from raw data to cleaned data"""

    def __init__(self, df, lang='en_core_web_lg', verbose=True):
        self.verbose = verbose
        self.df = df
        try:
            self.nlp = spacy.load(lang)
        except OSError:
            raise Exception('It seems like the spacy corpus {lang} is not installed! You can install it using:\npython -m spacy download {lang}'.format(lang=lang))
        self.logger = logging.getLogger(__name__)

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
        text = self.df['text'].apply(str)
        # remove user handles and URLs from text
        text = text.apply(lambda t: re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', t))
        text = text.apply(lambda t: re.sub('(\@[^\s]+)', '', t))
        text = text.apply(lambda t: self.nlp(t, disable=['parser', 'tagger', 'ner']))
        # Count the number of tokens excluding stopwords
        self.df['token_count'] = text.apply(lambda doc: len([token for token in doc if token.is_alpha and not token.is_stop]))

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

def run(dtypes=['original'], lang='en', verbose=False):
    s_time = time.time()
    logger = logging.getLogger(__name__)
    for dtype in dtypes:
        logger.info('Reading data of type {}...'.format(dtype))
        df = get_merged_data(dtype=dtype)
        num_tweets_with_duplicates = len(df)
        clt = CleanTweets(df, lang=lang, verbose=verbose)
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
