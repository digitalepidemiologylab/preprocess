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
from utils.helpers import get_parsed_data, find_project_root
import uuid

logger = logging.getLogger(__name__)

def run(args):
    # keep track of lengths
    df_len = {}

    # Load data
    logger.info('Reading raw data...')
    df = get_parsed_data(num_files=10, usecols=['id', 'text', 'lang', 'is_retweet'])
    logger.info(f'... read a total of {len(df):,} tweets')
    df.reset_index(drop=True, inplace=True)
    df_len['0_read'] = len(df)

    min_num_tokens = 5
    min_num_chars = 10
    keywords = ['masks', 'face covers', 'n95', 'kn95', 'ffp2']
    lang = 'en'
    exec_time = datetime.now().timestamp()

    # Take a sample
    num_raw_samples = len(df)
    df = df.sample(frac=.2, random_state=0)

    # Remove retweets
    logger.info('Removing retweets...')
    df = df[~df.is_retweet]
    logger.info(f'... {len(df):,} remaining')
    df_len['1_remove_retweets'] = len(df)

    # Select English tweets
    logger.info(f'Filter by lang {lang}...')
    df = df[df.lang==lang]
    logger.info(f'... {len(df):,} remaining')
    df_len['2_remove_lang'] = len(df)

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

    # Select tweets based on keyword matching
    logger.info(f'Filter by keywords {keywords}...')
    df = df[df.text.str.contains(r'\bmasks?\b|\bface covers?\b|\bn95\b|\bkn95\b|\bffp2?\b')]
    logger.info(f'... {len(df):,} remaining')
    df_len['3_keep_keywords'] = len(df)
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
    df_len['4_remove_min_num_tokens'] = len(df)

    # Remove tweets with less than 10 characters
    logger.info(f'Filter by min_num_chars {min_num_chars}...')
    df = df[df.text_cleaned.str.len() > min_num_chars]
    logger.info(f'... {len(df):,} remaining')
    df_len['5_remove_min_num_chars'] = len(df)
    
    #Empty array to be filled with indices corresponding to near-duplicates
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
    logger.info('Drop near-duplicates...')
    df = df.drop_duplicates(subset=['text_cleaned'])
    logger.info(f'... {len(df):,} remaining')
    df_len['6_remove_near-duplicates'] = len(df)

    # Write sample file
    # f_out_folder = os.path.join(find_project_root(), 'data', '2_sampled')
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


if __name__ == '__main__':
    run({})
