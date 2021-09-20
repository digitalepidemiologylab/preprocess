import pandas as pd
from pathlib import Path
import numpy as np
import os
from collections import defaultdict
import glob
import ast
import time
import json
from tqdm import tqdm
import multiprocessing
import joblib
import logging
from utils.misc import file_lock
from utils.helpers import get_project_info, get_dtypes
from utils.process_tweet import ProcessTweet
from datetime import datetime
from utils.geo_helpers import load_map_data
from geocode.geocode import Geocode
import shutil
import pickle
import gzip
import bz2
import ray
import sys


logger = logging.getLogger(__name__)
output_folder = os.path.join('data', '1_parsed')
input_folder = os.path.join('data', '0_raw')

manager = multiprocessing.Manager()
# shared between all processes
originals = manager.dict()
retweet_counts = manager.dict()
quote_counts = manager.dict()
replies_counts = manager.dict()

extract_retweets=True
extract_quotes=True

def generate_file_list():
    """generates dictionary of files per day"""
    f_names = []
    for file_type in ['jsonl', 'gz', 'bz2']:
        globbed = Path(input_folder).rglob(f'tweets-20210206*.{file_type}')
        f_names.extend([str(g) for g in globbed])
    grouped_f_names = defaultdict(list)
    datefmt = '%Y-%m-%d'
    for f_name in f_names:
        if os.path.basename(f_name).startswith('tweets'):
            date_str = f_name.split('-')[1]
            day_str = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime(datefmt)
        elif os.path.basename(f_name).startswith('crowdbreaks'):
            date_str = re.findall(r'\d{4}\-\d+\-\d+\-\d+\-\d+\-\d+', os.path.basename(f_name))[0]
            day_str = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S').strftime(datefmt)
        else:
             day_str = '-'.join(f_name.split('/')[-5:-2])
        grouped_f_names[day_str].append(f_name)
    return grouped_f_names

def extract_tweets(day, f_names, keywords):
    gc = Geocode()
    gc.load()
    map_data = load_map_data()
    # create dirs
    for subfolder in ['tweets', 'preliminary']:
        folder = os.path.join(output_folder, subfolder)
        if not os.path.isdir(folder):
            os.makedirs(folder)
    def write_to_file(obj):
        created_at_day = obj['created_at'][:10]
        f_path = os.path.join(output_folder, 'preliminary', f'{created_at_day}.jsonl')
        with open(f_path, 'a') as f_out:
            with file_lock(f_out):
                f_out.write(json.dumps(obj) + '\n')
    for f_name in f_names:
        if f_name.endswith('.gz'):
            f = gzip.open(f_name, 'r')
        elif f_name.endswith('.bz2'):
            f = bz2.BZ2File(f_name, 'r')
        else:
            f = open(f_name, 'r')
        for i, line in enumerate(f):
            if len(line) <= 1:
                continue
            try:
                tweet = json.loads(line)
            except json.decoder.JSONDecodeError:
                if isinstance(line, bytes):
                    # gzip.open will provide file in bytes format, in this case deocde
                    line = line.decode()
                # some files use single quotation (Python dictionaries), for this we need to use ast.literal_eval
                tweet = ast.literal_eval(line)
            except:
                # sometimes parsing completely fails
                logger.error('Error parsing line:')
                logger.error(line)
                continue
            if not 'id' in tweet:
                continue  # API logs, error messages, etc.
            tweet_id = tweet['id_str']
            if tweet_id in originals:
                # skip duplicates
                continue
            # flag tweet ID as "used"
            originals[tweet_id] = True
            # extract tweet
            pt = ProcessTweet(tweet=tweet, keywords=keywords, map_data=map_data, gc=gc)
            if ((extract_retweets and pt.is_retweet)                              # extract retweets (optional)
                    or (extract_quotes and pt.has_quote and not pt.is_retweet)    # extract quotes if not retweet of a quote (optional)
                    or (not pt.is_retweet and not pt.has_quote)):                 # always extract original tweets which are neither retweets nor quotes
                extracted_tweet = pt.extract()
                write_to_file(extracted_tweet)
            # add interaction counts
            if pt.is_reply:
                if pt.replied_status_id in replies_counts:
                    replies_counts[pt.replied_status_id] += 1
                else:
                    replies_counts[pt.replied_status_id] = 1
            if pt.has_quote:
                pt_quote = ProcessTweet(tweet=tweet['quoted_status'], keywords=keywords, map_data=map_data, gc=gc)
                if not pt.is_retweet:
                    if pt_quote.id in quote_counts:
                        quote_counts[pt_quote.id] += 1
                    else:
                        quote_counts[pt_quote.id] = 1
                if not pt_quote.id in originals:
                    # extract original status
                    originals[pt_quote.id] = True
                    extracted_tweet = pt_quote.extract()
                    write_to_file(extracted_tweet)
            if pt.is_retweet:
                pt_retweet = ProcessTweet(tweet=tweet['retweeted_status'], keywords=keywords, map_data=map_data, gc=gc)
                if pt_retweet.id in retweet_counts:
                    retweet_counts[pt_retweet.id] += 1
                else:
                    retweet_counts[pt_retweet.id] = 1
                if pt_retweet.id not in originals:
                    # extract original status
                    originals[pt_retweet.id] = True
                    extracted_tweet = pt_retweet.extract()
                    write_to_file(extracted_tweet)
        f.close()

def main():
    project_info = get_project_info()
    keywords = project_info['keywords']

    grouped_f_names = generate_file_list()
    num_cores = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cores)
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    parallel(extract_tweets_delayed(key, f_names, keywords) for key, f_names in tqdm(grouped_f_names.items()))

if __name__ == '__main__':
    main()
