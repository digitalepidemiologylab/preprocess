#!/usr/bin/env python
# coding: utf-8

import pdb
import pandas as pd
import numpy as np
from utils.helpers import get_parsed_data
import unicodedata
import emoji
from html.parser import HTMLParser
# Import module for regular expressions
import re
# Import Python extension for computing string edit distances and similarities
import Levenshtein as lev

import sys
sys.path.append('/drives/sde/wuhan_project/preprocess') # data until June 7, 2020
# sys.path.append('/drives/sdf/martin/preprocess') # data until September 14, 2020


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

# Load data
df = get_parsed_data(num_files=10, usecols=['id', 'text', 'lang', 'is_retweet'])
df.reset_index(drop=True, inplace=True)
df

# Take a sample
sample_df = df.sample(frac=0.001, random_state=0)

# Remove retweets
sample_df = sample_df.loc[sample_df['is_retweet']!=True]

# Select English tweets
sample_df = sample_df.loc[sample_df['lang']=='en'].copy()
#Compute the number of remaining tweets
len_sample = len(sample_df)
print('Total number of English tweets: ', len_sample)

# **Cleaning the data**
# Create an instance of a HTML parser
html_parser = HTMLParser()
# Escape HTML symbols
sample_df.text = sample_df.text.apply(html_parser.unescape)

# Replace suspension points with the standard dots
sample_df.text = sample_df.text.str.replace('…','...')

# Normalize Unicode characters
sample_df.text = sample_df.text.map(lambda x: unicodedata.normalize('NFKC', x))

# Erase usernames
sample_df.text = sample_df.text.str.replace(r'(^|[^@\w])@(\w{1,15})\b', '')

# Erase URLs
sample_df.text = sample_df.text.str.replace(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','')

# Replace \t, \n and \r characters with a whitespace
sample_df.text = sample_df.text.str.replace(r'[\r\n\t]+', ' ')


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


# **Select the potentially relevant tweets**

# Remove gibberish
sample_df = sample_df.loc[sample_df.text.str.len()>5].copy()

# Compute the number of remaining tweets
len_sample = len(sample_df)
print('Total number of potentially relevant English tweets: ', len_sample)

# **Removing near-duplicates and selecting tweets based on keyword matching**

#Empty array to be filled with indices corresponding to near-duplicates
dup_idx = []
for i in range(len_sample-1):
	for j in range(i+1, len_sample):
		# Identify near-duplicates within sample_df		
		if lev.distance(sample_df.text.iloc[i], sample_df.text.iloc[j])<10:
		# Add new near-duplicates to the array dup_idx	
			if j not in dup_idx:
				dup_idx.append(j)
# Remove near-duplicates
sample_df = sample_df.drop(labels=sample_df.iloc[dup_idx].index)

# Compute the number of remaining tweets
len_sample = len(sample_df)
print('Number of remaining tweets before keyword matching: ' , len_sample)

# Select tweets based on keyword matching
keyword_bool = sample_df.text.str.contains(r'wear|masks?|protect|\bppe\b|\bnpi\b|\bn95\b|\bkn95\b|\bffp2?\b')
clean_sample = sample_df[keyword_bool]
print('Number of relevant tweets: ',len(clean_sample))

