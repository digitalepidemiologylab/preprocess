#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import joblib
import multiprocessing
import unicodedata
import emoji
from html.parser import HTMLParser
# Import module for regular expressions
import re
# Import Python extension for computing string edit distances and similarities
import Levenshtein as lev

import uuid
from tqdm import tqdm
import pdb

import os
from datetime import datetime
import time
import sys
import glob
import logging
sys.path.append('/drives/sde/wuhan_project/preprocess') # data until June 7, 2020
# sys.path.append('/drives/sdf/martin/preprocess') # data until September 14, 2020
# sys.path.append('../..')
from utils.helpers import get_parsed_data, get_sampled_data, get_labelled_data, get_cleaned_labelled_data, get_uploaded_batched_data, get_batched_sample_data, find_folder, #find_project_root 
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
        """Generates a new batch which is randomly sampled from the cleaned data"""
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
            self.logger.warn('No sample files written. Aborting.')
            return
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        min_date_str = ''
        if min_date is not None:
            min_date_str = '_min_date_{}'.format(min_date)
        max_date_str = ''
        if max_date is not None:
            max_date_str = '_max_date_{}'.format(max_date)
        
        # Write a CSV file
        data_f_name = 'sampled_{mode}_{len_sample}_{size}_{seed}{min_date}{max_date}_created_{timestamp}{flags}_data.csv'.format(mode=mode, len_sample=len(sample),
                size=size, seed=self.seed, timestamp=timestamp, min_date=min_date_str, max_date=max_date_str, flags=flags)
        data_full_path = os.path.join(find_folder('2_sampled'), data_f_name)
        self.logger.info('Writing CSV file {} ...'.format(data_full_path))
        if 'all' in columns:
            sample.to_csv(data_full_path, encoding='utf8')
        else:
            sample[columns].to_csv(data_full_path, encoding='utf8', index=False, header=False)

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

        #stats
        still_available = tweet_ids_sampled - tweet_ids_unavailable - tweet_ids_labelled
        if not ignore_previous:
            still_available -= tweet_ids_batched
        logger.info('Unique tweets in base sample(s): {:,} (labelled: {:,}, unavailable: {:,}, in previous batches: {:,})'.format(len(tweet_ids_sampled), len(tweet_ids_labelled), len(tweet_ids_unavailable), len(tweet_ids_batched)))
        logger.info('Tweets left to sample from: {:,}'.format(len(still_available)))
        logger.info('Percentage labelled: {:.2f}%'.format(100*float(len(tweet_ids_labelled)/len(tweet_ids_sampled))))

    def generate_batch(self, num_tweets=None, batch_id=None, tail=True, ignore_previous=False):
        """ Generates a new batch which takes as input a large sample file provided in `data/2_sampled` and generates a new batch not including previously annotated tweets. """

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
        logger.info('Percentage labelled: {:.2f}%'.format(100*float(len(tweet_ids_labelled)/len(tweet_ids_sampled))))
        # return conditions
        if len(still_available) <= 0:
            logger.warn('All available tweets have been labelled'.format(len(tweet_ids_sampled), len(still available)))
            return
        if num_tweets > len(still_available):
            logger.warn('Requested to create batch of {:,}, but only {:,} are still available'.format(num_tweets, len(still_available)))
            return
        if tail:
            batch = df_samples.loc[df_samples['tweet_id'].isin(still_available)][-num:tweets:]
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
            raise Exception('Found pre-existing folder "{}". Please remove this folder first or pick a different batch ID'.format(output_folder))
        f_path = os.path.join(output_folder, '{}_{}.csv'.format(batch_name, datetime.now().strftime('%Y-%m-%d')))
        batch.to_csv(f_path, header=None, index=False, encoding='utf8')
        logger.info('Successfully wrote file containing new batch "{}"'.format(f_path))


    # methods for distribution analysis

    def monthdelta(self, date, base):
        year_delta = int(date[0:4]) - int(base[0:4])
        month_delta = int(date[5:7]) - int(base[5:7])
        return year_delta * 12 + month_delta

    def create_distributions(self, df=None, sample=None):
        if df is None:
            df = self.df
        if sample is None:
            sample = self.sample
        df['in_sample'] = df['id'].isin(sample['id'])
        df['datetime'] = [str(idx) for idx in df.index]
        base_date_str = df.datetime[0][0:10]
        base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
        df['day_idx'] = [(datetime.strptime(str(idx[0:10], '%Y-%m-%d') - base_date)]
        df['month_idx'] = [self.monthdelta(str(idx)[0:10], base_date_str) for idx in idx in df.datetime]
        df['idx'] = range(0, len(df))

        self.indices = df.loc[df['in_sample']]['idx']
        self.days = df.loc[df['in_sample']]['day_idx']
        self.months = df.loc[df['in_sample']]['month_idx']
        self.years = df.loc[df['in_sample']]['year_idx']
        return self.indices, self.days, self.months, self.years 


# def run(args):
def run(size=None, langs=None, include_replies=False, anonymize=True, contains_keywords=False, min_token_count=5, min_char_count=10, mode='monthly', seed=None, extend=False, bin_size=None, min_date=None, max_date=None): 
args_dict = {'size': size, 'langs': langs, 'include_replies': include_replies, 'anonymize': anonymize, 
            'contains_keywords': contains_keywords, 'min_token_count': min_token_count, 'min_char_count': min_char_count,
            'mode': mode, 'seed': seed, 'extend': extend, 'bin_size': bin_size, 'min_date': min_date, 'max_date': max_date}
    if bin_size is None:
        logger.info('Creating sample of size {:,}...'.format(size))
    else:
        logger.info('Creating sample of size {:,} or bin size {:,}'.format(size, bin_size))
     
    # Load data
    df = get_parsed_data(num_files=10, usecols=['id', 'text', 'created_at', 'lang', 'is_reply', 'is_retweet'],
            contains_keywords=contains_keywords,
            num_files=200,s_date=min_date, e_date=max_date)
    
    flags = ''
    df.reset_index(drop=True, inplace=True)

    # Keep track of lengths
    df_len = {}
    df_len['0_read'] = len(df)

    logger.info(f'Read a total of {len(df):,} tweets. Filtering...')
    
    # Filter retweets
    logger.info('Filtering retweets...')
    df = df[~df.is_retweet]
    logger.info(f'... {len(df):,} remaining')
    df_len['1_remove_retweets'] = len(df)
    
    # Filter by date
    if min_date is not None or max_date is not None:
        logger.info('Filtering by dates...')
        df = df.set_index('created_at')[min_date:max_date].reset_index()
        df_len['2_date'] = len(df)
        logger.info(f'... {len(df):,} remaining')


    if not include_replies:
        # by default filter replies
        logger.info('Filtering replies...')
        df = df[~df.is_reply]
        df_len['3_without_replies'] = len(df)
        logger.info(f'... {len(df):,} remaining')
    else:
        logger.info('Including replies...')
        flags += '_include_replies'

    # Filter by language
    if isinstance(langs, list):
        if len(langs) > 0:
            logger.info('Filtering for languages {}...'.format(', '.join(langs))
            df = df[df.lang.isin(langs)]
            flags += '_langs_{}'.format(', '.join(langs))
            df_len['4_lang'] = len(df)
            logger.info(f'... {len(df):,} remaining')

    # Filter previous
    if extend:
        logger.info('Extending previous sampled data...')
        flags += '_extended'
        df_sampled = get_sampled_data()
        df = df[~df.id.isin(df_sampled.tweet_id)]
        df = df[~df.text.isin(df_sampled.tweet_text)]
    
    # def word_len(word_arr):
    #     return len(word_arr) > 4
    # Remove tweets with less than 5 tokens
    # parallel = joblib.Parallel(n_jobs=8)
    # boolean_l = parallel(joblib.delayed(word_len)(df.text.str.split().iloc[i]) for i in range(len(df)))
    # df = df[boolean_l].copy()

    # General cleaning of data
    # Create an instance of a HTML parser
    html_parser = HTMLParser()

    # Escape HTML symbols
    df.text = df.text.apply(html_parser.unescape)

    # Replace suspension points with the standard dots
    df.text = df.text.str.replace('…','...')

    # Normalize Unicode characters
    df.text = df.text.map(lambda x: unicodedata.normalize('NFKC', x))

    # Anonymize: erase URLs and usernames
    if anonymize:
        # Erase usernames
        df.text = df.text.str.replace(r'(^|[^@\w])@(\w{1,15})\b', '')
    
        # Erase URLs
        df.text = df.text.str.replace(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','')

    # Create copy of text column
    df['text_cleaned'] = df['text']
   
    # Modifications of the text enabling the identification of duplicates and near-duplicates
    def misc_rem(text):
        # Remove all emojis
        text = ''.join('' if unicodedata.category(c)[0] == 'S' else c for c in text)

        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        text = text.strip()
        
        # Transform to lower-case characters
        text = text.lower()
        return text
        
    df['text_cleaned'] = df.text_cleaned.apply(misc_rem)

    # Before removing duplicates and near-duplicates, apply additional filters (number of tokens, number of characters, presence of keywords)
    # Filter tweets with less than 5 tokens
    logger.info(f'Filter by min_num_tokens {min_num_tokens}...')
    df['token_count'] = df.text_cleaned.apply(lambda s: len(s.split()))
    df = df[df.token_count > min_token_count]
    logger.info(f'... {len(df):,} remaining')
    df_len['5_token_count'] = len(df)

    # Filter tweets with less than 10 characters
    logger.info(f'Filter by min_num_chars {min_num_chars}...')
    df = df[df.text_cleaned.str.len() > min_char_count]
    logger.info(f'... {len(df):,} remaining')
    df_len['6_char_count'] = len(df)

    # Store the value of contain_keywords passed to get_parsed_data
    if contains_keywords:
        # data was already filtered if contains_keywords was set to True (default: False) in get_parsed_data
        logger.info('Filtered for contains_keywords...')
        flags += '_contains_keywords'
   
    # Filter tweets by keywords
    keywords = ['masks', 'face covers', 'n95', 'kn95', 'ffp2']
    logger.info('Filter by keywords {}...'.format(', '.join(keywords)))
    for i in range(len(keywords)-1);
        keywords[i] = r'\b' + keywords[i] + r'\b|'
    keywords[-1] = r'\b' + keywords[-1] + r'\b'
    kw_regex = ''.join(keywords)
    df = df[df.text.str.contains(kw_regex)]
    df_len['7_keywords'] = len(df)
    logger.info(f'... {len(df):,} remaining')

    # Drop duplicates
    logger.info('Dropping duplicates...')
    df = df.drop_duplicates(subset=['text_cleaned'])
    logger.info(f'... {len(df):,} remaining')
    df_len['8_duplicates'] = len(df)

    # Empty array to be filled with indices corresponding to near-duplicates
    def str_combination_gen(dfr):
        for i in range(len(dfr)):
            for j in range(i+1):
                yield (dfr.text.iloc[i],dfr.text.iloc[j])
        
    def lev_distance(text1,text2):
        bool_list = []
        # Identify near-duplicates within df		
        if lev.distance(text1, text2)<10:
            # Add new near-duplicates to the array dup_idx 
            bool_list.append(1)
        else: 
            bool_list.append(0)
        return bool_list 
    
    # idx_couples = str_combination_gen(df)
    # lev_distance_delayed = joblib.delayed(lev_distance)
    # parallel = joblib.Parallel(n_jobs = 8)
    # bool_list = parallel(lev_distance_delayed(*arg) for arg in idx_couples)
    # bool_arr = np.reshape(np.array(bool_list), len(bool_list))
    # l_arr = np.zeros((len_sample, len_sample))
    # indices = np.tril_indices(len_sample)
    # l_arr[indices] = bool_arr
    # l_df = pd.DataFrame(l_arr, index=df.index, columns=df.index)
    
    # # Indices that correspond to unique text entries
    # ndup_idx = l_df[l_df.sum(axis=0)==1].index
    # # Keep the original tweets
    # logger.info('Dropping near-duplicates...')
    # df = df.loc[ndup_idx]
    # logger.info(f'... {len(df):,} remaining')
    # df_len['9_nearduplicates'] = len(df)
    

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
    # if anonymize:
    #     logger.info('Anonymizing sample...')
    #     sample.loc[:, 'text'] = sample.text.apply(ProcessTweet.anonymize_text)
    
    # Write CSV file
    generator.write_sample(sample, mode, size=('bin' + str(bin_size)) if size is None else size, min_date=min_date, max_date=max_date, flags=flags)

    # Write config file
    if len(sample) != 0: 
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        min_date_str = ''
        if min_date is not None:
            min_date_str = '_min_date_{}'.format(min_date)
        max_date_str = ''
        if max_date is not None:
            max_date_str = '_max_date_{}'.format(max_date)
        
        config_f_name = 'sampled_{mode}_{len_sample}_{size}_{seed}{min_date}{max_date}_created_{timestamp}{flags}_config.json'.format(mode=mode, len_sample=len(sample),
                size=size, seed=self.seed, timestamp=timestamp, min_date=min_date_str, max_date=max_date_str, flags=flags)
        config_full_path = os.path.join(find_folder('2_sampled'), config_f_name)
        self.logger.info('Writing JSON file {} ...'.format(config_full_path))
        output_data = {**args_dict, **df_len, 'keywords': keywords, 'timestamp': timestamp}
        with open(f_out_config_path, 'w') as f:
            json.dump(output_data, f)

# if __name__ == '__main__':
#     run({})
