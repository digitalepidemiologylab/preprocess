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
import pdb
import logging
import os
import sys
from tqdm import tqdm
from datetime import datetime
sys.path.append('/drives/sde/wuhan_project/preprocess') # data until June 7, 2020
# sys.path.append('/drives/sdf/martin/preprocess') # data until September 14, 2020
from utils.helpers import get_parsed_data, get_sampled_data, get_labelled_data, get_cleaned_labelled_data, get_uploaded_batched_data, get_batched_sample_data, find_folder, #find_project_root 
import uuid

class SampleGenerator(object):
    """Class for generating samples from tweet corpus"""
    
    def __init__(self, seed=None):
        self.logger = logging.getLogger(__name__)
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

    def create_month_bins(self, df):
        return self.create_bins(df, 'months', 0, 7)
    
    def random_sample(self, df, size):
        """Generates a new batch which is randomly sampled from the cleaned data"""
        return df.sample(size, random_state=self.seed)
    
    def create_sample(self, bins=None, bins_unused=None, bins_used=None, bin_size=None):
        if bins is None:
            bins = self.bins
        bin_count = len(bins)
        if bin_size is None:
            bin_size = int(size/ bin_count)
        samples = []
        for unique in bins:

        sample = pd.concat(samples)
        self.sample = sample
        return sample

    def write_sample(self, sample, dtype, mode, columns=['id','text'], size='', min_date=None, max_date=None, flags=''):
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
        f_name = 'sampled_{dtype}_{mode}_{len_sample}_{size}_{seed}{min_date}{max_date}_created_{timestamp}{flags}.csv'.format(dtype=dtype, mode=mode, len_sample=len(sample),
                size=size, seed=self.seed, timestamp=timestamp, min_date=min_date_str, max_date=max_date_str, flags=flags)
        full_path = os.path.join(find_folder('2_sampled'), f_name)
        self.logger.info('Writing file {} ...'.format(full_path))
        if 'all' in columns:
            sample.to_csv(full_path, encoding='utf8')
        else:
            sample[columns].to_csv(full_path, encoding='utf8', index=False, header=False)

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
def run(dtype='anonymized', size=None, bin_size=None, langs='en', include_replies=False, mode='monthly', seed=None, extend=False, min_date=None, max_date=None):
    logger = logging.getLogger(__name__)
    
    if bin_size is None:
        logger.info('Creating sample of size {:,}...'.format(size))
    else:
        logger.info('Creating sample of size {:,} or bin size {:,}'.format(size, bin_size))
    
    logger.info('Reading data of type "{}"...'.format(dtype))
        
    # Load data
    df = get_parsed_data(num_files=10, usecols=['id', 'text', 'is_duplicate', 'created_at', 'use_for_labelling', 'lang', 'is_retweet', 'in_reply_to_status_id'])
    
    # Keep track of lengths
    df_len = {}
    df_len['0_read'] = len(df)

    logger.info(f'Read a total of {len(df):,} tweets. Filtering...')
    flags = ''

    df.reset_index(drop=True, inplace=True)
    min_num_tokens = 5
    min_num_chars = 10
    exec_time = datetime.now().timestamp()

    # Take a sample
    num_raw_samples = len(df)
    df = df.sample(frac=.2, random_state=0)
    
    # Filter by date
    if min_date is not None or max_date is not None:
        logger.info('Filtering by dates...')
        df = df[min_date:max_date]
        df_len['1_date'] = len(df)
        logger.info(f'... {len(df):,} remaining')

    # Select use for labelling (-> retweets and extracted tweets not included) and no duplicate
    logger.info('Filtering for default flags (use_for_labelling/is_duplicate)...')
    df = df[(df.use_for_labelling) & (~df.is_duplicate)]
    df_len['2_def_flags'] = len(df)
    logger.info(f'... {len(df):,} remaining')

    if not include_replies:
        # by default filter replies
        logger.info('Filtering replies...')
        df = df[df.in_reply_to_status_id.isna()]
        df_len['3_without_replies'] = len(df)
        logger.info(f'... {len(df):,} remaining')
    else:
        logger.info('Including replies...')
        flags += '_include_replies'

    # # Remove retweets
    # logger.info('Filtering retweets...')
    # df = df[~df.is_retweet]
    # logger.info(f'... {len(df):,} remaining')
    # df_len['1_remove_retweets'] = len(df)
    
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
    
    # General cleaning of data
    # Create an instance of a HTML parser
    html_parser = HTMLParser()

    # Escape HTML symbols
    df.text = df.text.apply(html_parser.unescape)

    # Replace suspension points with the standard dots
    df.text = df.text.str.replace('…','...')

    # Normalize Unicode characters
    df.text = df.text.map(lambda x: unicodedata.normalize('NFKC', x))

    # Erase usernames
    df.text = df.text.str.replace(r'(^|[^@\w])@(\w{1,15})\b', '')

    # Erase URLs
    df.text = df.text.str.replace(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','')

    # Filter by keywords
    keywords = ['masks', 'face covers', 'n95', 'kn95', 'ffp2']
    logger.info('Filter by keywords {}...'.format(', '.join(keywords)))
    for i in range(len(keywords)-1);
        keywords[i] = r'\b' + keywords[i] + r'\b|'
    keywords[-1] = r'\b' + keywords[-1] + r'\b'
    kw_regex = ''.join(keywords)
    df = df[df.text.str.contains(kw_regex)]
    df_len['5_keep_keywords'] = len(df)
    logger.info(f'... {len(df):,} remaining')
    
    # Create copy of text column
    df['text_cleaned'] = df['text']
    
    # Operations for filtering near-duplicates
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

    # def word_len(word_arr):
    #     # split
    #     return len(word_arr) > 4
    # Remove tweets with less than 5 words
    # parallel = joblib.Parallel(n_jobs=8)
    # boolean_l = parallel(joblib.delayed(word_len)(df.text.str.split().iloc[i]) for i in range(len(df)))
    # df = df[boolean_l].copy()

    logger.info(f'Filter by min_num_tokens {min_num_tokens}...')
    df['num_tokens'] = df.text_cleaned.apply(lambda s: len(s.split()))
    df = df[df.num_tokens > min_num_tokens]
    logger.info(f'... {len(df):,} remaining')
    df_len['6_remove_min_num_tokens'] = len(df)

    # Remove tweets with less than 10 characters
    logger.info(f'Filter by min_num_chars {min_num_chars}...')
    df = df[df.text_cleaned.str.len() > min_num_chars]
    logger.info(f'... {len(df):,} remaining')
    df_len['7_remove_min_num_chars'] = len(df)
    
    # Empty array to be filled with indices corresponding to near-duplicates
    def str_combination_gen(dfr):
        for i in range(len(dfr)):
            for j in range(i+1):
                yield (dfr.text.iloc[i],dfr.text.iloc[j])
        
    def lev_distance(text1,text2):
        bool_list=[]
        # Identify near-duplicates within df		
        if lev.distance(text1, text2)<10:
            # Add new near-duplicates to the array dup_idx 
            bool_list.append(1)
        else: 
            bool_list.append(0)
        return bool_list 
    
    #idx_couples = str_combination_gen(df)
    #lev_distance_delayed = joblib.delayed(lev_distance)
    #parallel = joblib.Parallel(n_jobs = 8)
    #bool_list = parallel(lev_distance_delayed(*arg) for arg in idx_couples)
    #bool_arr = np.reshape(np.array(bool_list), len(bool_list))
    #l_arr = np.zeros((len_sample, len_sample))
    #indices = np.tril_indices(len_sample)
    #l_arr[indices] = bool_arr
    #l_df = pd.DataFrame(l_arr, index=df.index, columns=df.index)
    
    # Indices that correspond to unique text entries
    #ndup_idx = l_df[l_df.sum(axis=0)==1].index
    # Keep the original tweets
    #df = df.loc[ndup_idx]
    
    # Drop duplicates
    logger.info('Dropping near-duplicates...')
    df = df.drop_duplicates(subset=['text_cleaned'])
    logger.info(f'... {len(df):,} remaining')
    df_len['8_remove_near-duplicates'] = len(df)

    # Write sample file
    # f_out_folder = os.path.join('..', find_project_root(), 'data', '2_sampled')
    f_out_folder = os.path.join('..', '..', 'data', '2_sampled')
    random_hash = str(uuid.uuid4())[:8]

    f_name = f'sampled_{random_hash}_data.csv'
    # df[['id', 'text']].to_csv(os.path.join(f_out_folder, f_name), index=False)
    df[['id', 'text']].to_csv(os.path.join(f_out_folder,f_name), index=False)

    output_data = {**args, **df_len, 'keywords': keywords, 'lang': lang, 'min_num_chars': min_num_chars, 'min_num_tokens': min_num_tokens, 'timestamp': exec_time}
    # add length after every cleaning step
    # all args (keywords, lang, ...)
    # add current timestamp

    # f_out_config_path = os.path.join(f_out_folder, f'sample_{random_hash}_config.json')
    f_out_config_path = os.path.join(f_out_folder, f'sampled_{random_hash}_config.json')
    with open(f_out_config_path, 'w') as f:
        json.dump(output_data, f)

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
    generator.write_sample(sample, dtype, mode, size=('bin' + str(bin_size)) if size is None else size, min_date=min_date, max_date=max_date, flags=flags)
# if __name__ == '__main__':
#     run({})
