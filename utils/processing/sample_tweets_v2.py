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

import sys
sys.path.append('/drives/sde/wuhan_project/preprocess') # data until June 7, 2020
# sys.path.append('/drives/sdf/martin/preprocess') # data until September 14, 2020

from utils.helpers import get_parsed_data

# Available columns:
# id                                     object
# text                                   object
# in_reply_to_status_id                  object
# in_reply_to_user_id                    object
# quoted_user_id                         object
# quoted_status_id                       object
# retweeted_user_id                      object
# retweeted_status_id                    object
# created_at                datetime64[ns, UTC]
# entities.user_mentions                 object
# user.id                                object
# user.screen_name                       object
# user.name                              object
# user.description                       object
# user.timezone                          object
# user.location                          object
# user.num_followers                      int64
# user.num_following                      int64
# user.created_at           datetime64[ns, UTC]
# user.statuses_count                     int64
# user.is_verified                         bool
# lang                                 category
# token_count                             int64
# is_retweet                               bool
# has_quote                                bool
# is_reply                                 bool
# contains_keywords                        bool
# longitude                             float64
# latitude                              float64
# country_code                         category
# region                               category
# subregion                            category
# geo_type                             category
# num_quotes                              int64
# num_replies                             int64
# num_retweets                            int64

def cleaning_f(num_files=10, frac=0.02, sel_lang='en', usecols=['id', 'text', 'lang', 'is_retweet', 'user.description']):
    # Load data
    df = get_parsed_data(num_files=num_files, usecols=usecols)
    df.reset_index(drop=True, inplace=True)
    cleanf_dict = {'language': sel_lang, 'keywords': ['masks', 'respirators', 'ppe', 'npi', 'n95', 'kn95', 'ffp2'],'min_char_len': 10, 'min_word_len': 5}

    # Take a sample
    sample_df = df.sample(frac=frac, random_state=0)
    num_raw_samples = len(df)

    # Remove retweets
    sample_df = sample_df.loc[sample_df['is_retweet']!=True]
    
    # Select English tweets
    sample_df = sample_df.loc[sample_df['lang']==sel_lang].copy()
    #Compute the number of remaining tweets
    len_sample = len(sample_df)
    print('Total number of English tweets: ', len_sample)
    
    # **Cleaning the data**
    # Create an instance of a HTML parser
    html_parser = HTMLParser()
    # Escape HTML symbols
    sample_df.text = sample_df.text.apply(html_parser.unescape)
    sample_df['user.description'] = sample_df['user.description'].apply(html_parser.unescape)

    # Replace suspension points with the standard dots
    sample_df.text = sample_df.text.str.replace('…','...')
    sample_df['user.description'] = sample_df['user.description'].str.replace('…','...')

    # Normalize Unicode characters
    sample_df.text = sample_df.text.map(lambda x: unicodedata.normalize('NFKC', x))
    sample_df['user.description'] = sample_df['user.description'].map(lambda x: unicodedata.normalize('NFKC', x))

    # Erase usernames
    sample_df.text = sample_df.text.str.replace(r'(^|[^@\w])@(\w{1,15})\b', '')
    sample_df['user.description'] = sample_df['user.description'].replace(r'(^|[^@\w])@(\w{1,15})\b', '')

    # Erase URLs
    sample_df.text = sample_df.text.str.replace(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','')
    sample_df['user.description'] = sample_df['user.description'].str.replace(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','')

    # Replace \t, \n and \r characters with a whitespace
    sample_df.text = sample_df.text.str.replace(r'[\r\n\t]+', ' ')
    sample_df['user.description'] = sample_df['user.description'].str.replace(r'[\r\n\t]+', ' ') 
    
    def misc_rem(text):
        # Remove all emojis
        text = ''.join('' if unicodedata.category(c)[0] == 'S' else c for c in text)
        # Remove all remaining control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        text = text.strip()
        
        # Transform to lower-case characters
        text = text.lower()
        
        return text
        
    sample_df.text = sample_df.text.apply(misc_rem)
    sample_df['user.description'] = sample_df['user.description'].apply(misc_rem)

    # **Select the potentially relevant tweets**
    
    # Remove tweets with less than 5 words
    boolean_l = [len(sample_df.text.str.split().iloc[i]) > 4 for i in range(len(sample_df))]
    sample_df = sample_df[boolean_l].copy()
    # Remove tweets with less than 11 characters
    sample_df = sample_df.loc[sample_df.text.str.len()>10].copy()
    
    # Compute the number of remaining tweets
    len_sample = len(sample_df)
    print('Total number of potentially relevant English tweets: ', len_sample)
    
    # **Removing near-duplicates and selecting tweets based on keyword matching**
    
    #Empty array to be filled with indices corresponding to near-duplicates
    
    def str_combination_gen(dfr):
        for i in range(len(dfr)):
            for j in range(i+1):
                yield (dfr.text.iloc[i],dfr.text.iloc[j])
        
    def lev_distance(text1,text2):
        bool_list=[]
        # Identify near-duplicates within sample_df		
        if lev.distance(text1, text2)<10:
            # Add new near-duplicates to the array dup_idx 
            bool_list.append(1)
        else: 
            bool_list.append(0)
        return bool_list 
    
    #idx_couples = str_combination_gen(sample_df)
    #lev_distance_delayed = joblib.delayed(lev_distance)
    #parallel = joblib.Parallel(n_jobs = 8)
    #bool_list = parallel(lev_distance_delayed(*arg) for arg in idx_couples)
    #bool_arr = np.reshape(np.array(bool_list), len(bool_list))
    #l_arr = np.zeros((len_sample, len_sample))
    #indices = np.tril_indices(len_sample)
    #l_arr[indices] = bool_arr
    #l_df = pd.DataFrame(l_arr, index=sample_df.index, columns=sample_df.index)
    
    # Indices that correspond to unique text entries
    #ndup_idx = l_df[l_df.sum(axis=0)==1].index
    # Keep the original tweets
    #sample_df = sample_df.loc[ndup_idx]
    
    # Compute the number of remaining tweets
    len_sample = len(sample_df)
    print('Number of remaining tweets before keyword matching: ' , len_sample)
    
    # Select tweets based on keyword matching
    keyword_bool = sample_df.text.str.contains(r'masks?|respirators?\b|\bppe\b|\bnpi\b|\bn95\b|\bkn95\b|\bffp2?\b')
    clean_sample = sample_df[keyword_bool]
    final_len = len(clean_sample)
    print('Number of relevant tweets: ',final_len)
    # Write sample file
    clean_sample.to_csv('../../data/2_sampled/sample_fp261020.csv')
    output_data = {**cleanf_dict, 'num_raw_samples':num_raw_samples, 'final_len': final_len}
    
    with open('config_cleaning.json', 'w') as file:
        json.dump(output_data, file)

    return

if __name__ == '__main__':
    cleaning_f()
