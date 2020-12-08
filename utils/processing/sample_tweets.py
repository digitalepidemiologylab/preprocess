import sys; sys.path.append('../..')
from utils.helpers import get_parsed_data, get_sampled_data, get_labelled_data, get_cleaned_labelled_data, find_folder, get_uploaded_batched_data, get_batched_sample_data
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
from random import randint
import sys
import glob
import logging
from utils.process_tweet import ProcessTweet

logger = logging.getLogger(__name__)

class SampleGenerator(object):
    """Class for generating samples from tweet corpus"""

    def __init__(self, seed=None):
        if seed is None:
            self.seed = randint(0,2**32-1)
        else:
            self.seed = seed

    def create_bins(self, df, column, lower, upper):
        self.df = df
        bins = {}
        _u = df.index.astype(str)
        df[column] = [i[lower:upper] for i in _u]
        for unique in df[column].unique():
            bins[unique] = df.loc[df[column] == unique]
        self.bins = bins
        self.bin_type = column
        return bins, column

    def create_all_bins(self, df):
        return self.create_bins(df, 'all', 0, 0)

    def create_year_bins(self, df):
        return self.create_bins(df, 'years', 0, 4)

    def create_month_bins(self, df):
        return self.create_bins(df, 'months', 0, 7)

    def create_day_bins(self, df):
        return self.create_bins(df, 'days', 0, 10)

    def random_sample(self, df, size):
        """Generates a new batch which is randomly sampled from the cleaned data
        """
        return df.sample(size, random_state=self.seed)

    def create_sample(self, bins=None, size=None, bins_unused=None, bins_used=None, bin_size=None):
        if bins is None:
            bins = self.bins
        bin_count = len(bins)
        if bin_size is None:
            bin_size = int(size / bin_count)
        samples = []
        for unique in bins:
            rows = bins[unique]
            target_size = bin_size
            exclude_unused = set()
            exclude_used = set()
            if not bins_unused is None and unique in bins_unused:
                rows_unused = bins_unused[unique]
                exclude_unused = set(rows_unused.id)
                target_size += len(exclude_unused)
            if not bins_used is None and unique in bins_used:
                rows_used = bins_used[unique]
                exclude_used = set(rows_used.id)
            if len(rows) < target_size:
                sample = rows.sample(frac=1)
            else:
                sample = rows.sample(n=target_size, random_state=self.seed)
            exclude = exclude_unused | exclude_used
            sample = sample[~sample.id.isin(exclude)]
            target_size = bin_size - len(exclude_used)
            if len(sample) > target_size:
                sample = sample[:target_size]
            samples.append(sample)
        sample = pd.concat(samples)
        self.sample = sample
        return sample

    def write_sample(self, sample, mode, columns=['id','text'], size='', min_date=None, max_date=None, flags=''):
        if len(sample) == 0:
            logger.warn('No sample files written. Aborting.')
            return
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        min_date_str = ''
        if min_date is not None:
            min_date_str = '_min_date_{}'.format(min_date)
        max_date_str = ''
        if max_date is not None:
            max_date_str = '_max_date_{}'.format(max_date)
        f_name = 'sampled_{mode}_{len_sample}_{size}_{seed}{min_date}{max_date}_created_{timestamp}{flags}.csv'.format(mode=mode, len_sample=len(sample),
                size=size, seed=self.seed, timestamp=timestamp, min_date=min_date_str, max_date=max_date_str, flags=flags)
        full_path = os.path.join(find_folder('2_sampled'), f_name)
        logger.info('Writing file {} ...'.format(full_path))
        if 'all' in columns:
            sample.to_csv(full_path, encoding='utf8')
        else:
            sample[columns].to_csv(full_path, encoding='utf8', index=False, header=False)

    def stats(self, ignore_previous=False):
        df_samples = get_sampled_data()
        try:
            df_labels = get_labelled_data()
        except FileNotFoundError:
            tweet_ids_labelled = set()
        else:
            tweet_ids_labelled = set(df_labels['tweet_id'])
        # Ids from previous batches
        df_batched = get_batched_sample_data()
        if len(df_batched) > 0:
            tweet_ids_batched = set(df_batched['tweet_id'])
        else:
            tweet_ids_batched = set()
        # Ids from previous batches which were not available
        df_unavailable = get_uploaded_batched_data(availability='unavailable')
        if len(df_unavailable) > 0:
            tweet_ids_unavailable = set(df_unavailable['tweet_id'])
        else:
            tweet_ids_unavailable = set()
        tweet_ids_sampled = set(df_samples['tweet_id'])
        # stats
        still_available = tweet_ids_sampled - tweet_ids_unavailable - tweet_ids_labelled
        if not ignore_previous:
            still_available -= tweet_ids_batched
        logger.info('Unique tweets in base sample(s): {:,} (labelled: {:,}, unavailable: {:,}, in previous batches: {:,})'.format(len(tweet_ids_sampled), len(tweet_ids_labelled), len(tweet_ids_unavailable), len(tweet_ids_batched)))
        logger.info('Tweets left to sample from: {:,}'.format(len(still_available)))
        logger.info('Precentage labelled: {:.2f}%'.format(100*float(len(tweet_ids_labelled)/len(tweet_ids_sampled))))


    def generate_batch(self, num_tweets=None, batch_id=None, tail=True, ignore_previous=False):
        """Generates a new batch which takes as input a large sample file provided in `data/2_sampled` and generates a new batch
        not including previously annotated tweets.
        """
        if num_tweets is None:
            raise ValueError('Num tweets is zero. Cannot create empty batch.')
        # vars
        sample_folder = find_folder('2_sampled')
        # Ids from sample file
        df_samples = get_sampled_data()
        if len(df_samples) == 0:
            raise Exception('Sample file is empty. Generate a sample file first.')
        tweet_ids_sampled = set(df_samples['tweet_id'])
        # Ids from previously labelled data
        try:
            df_labels = get_labelled_data()
        except FileNotFoundError:
            tweet_ids_labelled = set()
        else:
            tweet_ids_labelled = set(df_labels['tweet_id'])
        # Ids from previous batches
        df_batched = get_batched_sample_data()
        if len(df_batched) > 0:
            tweet_ids_batched = set(df_batched['tweet_id'])
        else:
            tweet_ids_batched = set()
        # Ids from previous batches which were not available
        df_unavailable = get_uploaded_batched_data(availability='unavailable')
        if len(df_unavailable) > 0:
            tweet_ids_unavailable = set(df_unavailable['tweet_id'])
        else:
            tweet_ids_unavailable = set()
        # remove tweets which are unavailable, have been previously labelled
        still_available = tweet_ids_sampled - tweet_ids_unavailable - tweet_ids_labelled
        if not ignore_previous:
            still_available -= tweet_ids_batched
        logger.info('Unique tweets in base sample(s): {:,} (labelled: {:,}, unavailable: {:,}, in previous batches: {:,})'.format(len(tweet_ids_sampled), len(tweet_ids_labelled), len(tweet_ids_unavailable), len(tweet_ids_batched)))
        logger.info('Tweets left to sample from: {:,}'.format(len(still_available)))
        logger.info('Precentage labelled: {:.2f}%'.format(100*float(len(tweet_ids_labelled)/len(tweet_ids_sampled))))
        # return conditions
        if len(still_available) <= 0:
            logger.warn('All available tweets have been labelled.'.format(len(tweet_ids_sampled), len(still_available)))
            return
        if num_tweets > len(still_available):
            logger.warn('Requested to create batch of {:,}, but only {:,} are still available.'.format(num_tweets, len(still_available)))
            return
        if tail:
            batch = df_samples.loc[df_samples['tweet_id'].isin(still_available)][-num_tweets:]
        else:
            batch = df_samples.loc[df_samples['tweet_id'].isin(still_available)][:num_tweets]
        assert len(batch) == num_tweets
        # write new batch file
        if batch_id is None:
            try:
                batch_id = 1 + max([int(s.split('_')[-1]) for s in os.listdir(sample_folder) if s.startswith('batch_') and os.path.isdir(os.path.join(sample_folder, s))])
            except ValueError:
                batch_id = 1
        batch_name = 'batch_{}'.format(batch_id)
        logger.info('Generating batch {} of size {:,} tweets...'.format(batch_name, num_tweets))
        output_folder = os.path.join(sample_folder, batch_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        else:
            raise Exception('Found pre-existing folder "{}". Please remove this folder first or pick a different batch ID.'.format(output_folder))
        f_path = os.path.join(output_folder, '{}_{}.csv'.format(batch_name, datetime.now().strftime('%Y-%m-%d')))
        batch.to_csv(f_path, header=None, index=False, encoding='utf8')
        logger.info('Successfully wrote file containing new batch "{}"'.format(f_path))

    # methods for distribution analysis

    def monthdelta(self,date,base):
        year_delta = int(date[0:4]) - int(base[0:4])
        month_delta = int(date[5:7]) - int(base[5:7])
        return year_delta * 12 + month_delta

    def create_distributions(self,df=None,sample=None):
        if df is None:
            df = self.df
        if sample is None:
            sample = self.sample
        df['in_sample'] = df['id'].isin(sample['id'])
        df['datetime'] = [str(idx) for idx in df.index]
        base_date_str = df.datetime[0][0:10]
        base_date = datetime.strptime(base_date_str,'%Y-%m-%d')
        df['day_idx'] = [(datetime.strptime(str(idx[0:10]),'%Y-%m-%d') - base_date).days for idx in df.datetime]
        df['month_idx'] = [self.monthdelta(str(idx)[0:10],base_date_str) for idx in df.datetime]
        df['year_idx'] = [int(idx[0:4]) - int(base_date_str[0:4]) for idx in df.datetime]
        df['idx'] = range(0,len(df))

        self.indices = df.loc[df['in_sample']]['idx']
        self.days = df.loc[df['in_sample']]['day_idx']
        self.months = df.loc[df['in_sample']]['month_idx']
        self.years = df.loc[df['in_sample']]['year_idx']
        return self.indices,self.days,self.months,self.years


def run(size=None, langs=None, include_replies=False, anonymize=True, contains_keywords=False, min_token_count=3, mode='monthly', seed=None, extend=False, bin_size=None, min_date=None, max_date=None):
    if bin_size is None:
        logger.info('Creating sample of size {:,}...'.format(size))
    else:
        logger.info('Creating sample of size {:,} or bin size {:,}...'.format(size, bin_size))
    df = get_parsed_data(usecols=['id', 'text', 'created_at', 'lang', 'is_reply', 'has_quote', 'token_count'],
            contains_keywords=contains_keywords,
            s_date=min_date,
            e_date=max_date)
    logger.info(f'Read {len(df):,} samples. Filtering...')
    flags = ''
    # Filter by date
    if min_date is not None or max_date is not None:
        logger.info('Filtering by dates...')
        df = df.set_index('created_at')[min_date:max_date].reset_index()
    # Min token count
    if isinstance(min_token_count, int):
        logger.info('Filtering by min_token_count...')
        df = df[df.token_count > min_token_count]
    if not include_replies:
        # by default filter replies
        df = df[~df.is_reply]
    else:
        logger.info('Including replies...')
        flags += '_include_replies'
    # Contains keywords
    if contains_keywords:
        # data was already filtered in get_parsed_data
        logger.info('Filtered for contains_keywords...')
        flags += '_contains_keywords'
    # Filter by language
    if isinstance(langs, list):
        if len(langs) > 0:
            logger.info('Filtering for languages {}...'.format(','.join(langs)))
            df = df[df.lang.isin(langs)]
            flags += '_langs_{}'.format(','.join(langs))
    # Filter previous
    if extend:
        logger.info('Extending previous sampled data...')
        flags += '_extended'
        df_sampled = get_sampled_data()
        df = df[~df.id.isin(df_sampled.tweet_id)]
        df = df[~df.text.isin(df_sampled.tweet_text)]
    # is_duplicate only marks duplicates before replacing <url> and @user tokens
    logger.info('Final screening for duplicates...')
    df['text_cleared'] = df.text.str.replace(r'@<user>|<url>', '')
    df['text_cleared'] = df.text_cleared.str.strip()
    df = df.drop_duplicates(subset=['text_cleared'])
    df = df.drop(['text_cleared'], axis=1) # release memory
    logger.info(f'... {len(df):,} rows in filtered data')
    generator = SampleGenerator(seed=seed)
    sample = pd.DataFrame()
    if mode == 'monthly':
        if extend:
            logger.info('Extending sample by evenly spread months based on seed "{}"...'.format(generator.seed))
            logger.info('Reading unavailable tweets...')
            df_unavailable = get_uploaded_batched_data('unavailable')
            unused_ids = set(df_unavailable['tweet_id'])
            unused = df[df.id.isin(unused_ids)].copy()
            unbins, _ = generator.create_month_bins(unused)
            logger.info('Reading available tweets...')
            df_available = get_uploaded_batched_data('available')
            used_ids = set(df_available['tweet_id'])
            used = pd.DataFrame(df[df.id.isin(used_ids)].copy())
            ubins, _ = generator.create_month_bins(used)
            logger.info('Generating sample...')
        else:
            unbins = None
            ubins = None
            logger.info('Generating sample by evenly spread months...')
        bins, bin_type = generator.create_month_bins(df)
        sample = generator.create_sample(bins, size=size, bins_unused=unbins, bins_used=ubins, bin_size=bin_size)
    elif mode == 'random':
        logger.info('Generating random sample...')
        sample = generator.random_sample(df, size)
    # anonymize
    if anonymize:
        logger.info('Anonymizing sample...')
        sample.loc[:, 'text'] = sample.text.apply(ProcessTweet.anonymize_text)
    generator.write_sample(sample, mode, size=('bin' + str(bin_size)) if size is None else size, min_date=min_date, max_date=max_date, flags=flags)
