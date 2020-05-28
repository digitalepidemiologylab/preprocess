import pandas as pd
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
from utils.helpers import get_project_info
from utils.process_tweet import ProcessTweet
from datetime import datetime
from utils.geo_helpers import load_map_data
from local_geocode.geocode.geocode import Geocode
import shutil


logger = logging.getLogger(__name__)
output_folder = os.path.join('data', '1_parsed')

def read_used_files():
    f_path = os.path.join(output_folder, f'.used_data')
    if not os.path.isfile(f_path):
        return {}
    with open(f_path, 'r') as f:
        used_files = json.load(f)
    return used_files

def write_used_files(data_files):
    f_path = os.path.join(output_folder, f'.used_data')
    with open(f_path, 'w') as f:
        json.dump(data_files, f, indent=4)

def generate_file_list():
    """generates dictionary of files per day"""
    output_path = os.path.join('data', '0_raw', 'streaming', '*json?')
    f_names = glob.glob(output_path)
    grouped_f_names = defaultdict(list)
    for f_name in f_names:
        if 'search_fullarchive' in f_name or 'search_30day' in f_name:
            date_str = f_name.split('_')[-1].split('.')[0]
        else:
            date_str = f_name.split('-')[1]
        day_str = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime('%Y-%m-%d')
        grouped_f_names[day_str].append(f_name)
    return grouped_f_names

def extract_tweets(day, f_names, project_info):
    gc = Geocode()
    gc.init()
    map_data = load_map_data()
    tweet_interaction_counts = defaultdict(lambda: {'num_quotes': 0, 'num_retweets': 0, 'num_replies': 0})
    f_out = os.path.join(output_folder, 'preliminary', f'{day}.jsonl')
    # create dirs
    for subfolder in ['tweets', 'preliminary']:
        folder = os.path.join(output_folder, subfolder)
        if not os.path.isdir(folder):
            os.makedirs(folder)
    def write_to_file(obj):
        with open(f_out, 'a') as f:
            f.write(json.dumps(obj) + '\n')
    collected_ids = set()
    for f_name in f_names:
        with open(f_name, 'r') as f:
            for i, line in enumerate(f):
                if len(line) <= 1:
                    continue
                try:
                    tweet = json.loads(line)
                except json.decoder.JSONDecodeError:
                    # some files use single quotation, for this we need to use ast.literal_eval
                    tweet = ast.literal_eval(line)
                except:
                    # sometimes parsing completely fails
                    logger.error('Error parsing line:')
                    logger.error(line)
                    continue
                if 'info' in tweet:
                    continue  # API log fields (can be ignored)
                if tweet['id_str'] in collected_ids:
                    # skip duplicates
                    continue
                pt = ProcessTweet(tweet=tweet, project_info=project_info, map_data=map_data, gc=gc)
                # add interaction counts
                if pt.is_reply:
                    tweet_interaction_counts[pt.replied_status_id]['num_replies'] += 1
                if pt.is_retweet:
                    tweet_interaction_counts[pt.retweeted_status_id]['num_retweets'] += 1
                if pt.has_quoted_status:
                    tweet_interaction_counts[pt.quoted_status_id]['num_quotes'] += 1
                # extract original tweet
                extracted_tweet = pt.extract(tweet_type='original')
                collected_ids.add(pt.id)
                write_to_file(extracted_tweet)
                if pt.is_retweet and tweet['retweeted_status']['id_str'] not in collected_ids:
                    # extract retweet
                    pt_retweet = ProcessTweet(tweet=tweet['retweeted_status'], project_info=project_info, map_data=map_data, gc=gc)
                    extracted_tweet = pt_retweet.extract(tweet_type='retweet')
                    write_to_file(extracted_tweet)
                    collected_ids.add(pt_retweet.id)
                if pt.has_quoted_status and tweet['quoted_status']['id_str'] not in collected_ids:
                    # extract quoted status
                    pt_quoted = ProcessTweet(tweet=tweet['quoted_status'], project_info=project_info, map_data=map_data, gc=gc)
                    extracted_tweet = pt_quoted.extract(tweet_type='quote')
                    write_to_file(extracted_tweet)
                    collected_ids.add(pt_quoted.id)
    return tweet_interaction_counts

def write_parquet_file(key, interaction_counts):
    df = []
    f_names = glob.glob(os.path.join(output_folder,  'preliminary', '*.jsonl'))
    # add interaction counts for every tweet
    for f_name in f_names:
        with open(f_name, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                created_at = tweet['created_at'].split('T')[0]
                if created_at == key:
                    tweet = {**tweet, **interaction_counts[tweet['id']]}
                    df.append(tweet)
    # drop duplicates
    df = pd.DataFrame(df)
    if len(df) > 0:
        df.drop_duplicates(subset=['id'], inplace=True)
        # convert columns to datetime
        for datetime_col in ['created_at', 'user.created_at']:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        # sort by created_at
        df.sort_values('created_at', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        # write parquet file
        f_out = os.path.join(output_folder, 'tweets', f'parsed_{key}.parquet')
        df.to_parquet(f_out)
    return len(df)

def merge_interaction_counts(res):
    """Merges all interaction from different days to a final dict of tweet_id->num_replies, num_quotes, num_retweets"""
    tweet_interaction_counts = defaultdict(lambda: {'num_quotes': 0, 'num_retweets': 0, 'num_replies': 0})
    for r in tqdm(res):
        for k, v in r.items():
            for _type in v.keys():
                tweet_interaction_counts[k][_type] += v[_type]
    return tweet_interaction_counts

def run(lang='en_core_web_sm', no_parallel=False):
    # setup
    s_time = time.time()
    project_info = get_project_info()

    # set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cores)

    # collect file list
    grouped_f_names = generate_file_list()

    # make sure map data is downloaded and geocode is set up
    map_data = load_map_data()
    gc = Geocode()
    gc.prepare()

    # run
    logger.info('Extract tweets...')
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    res = parallel((extract_tweets_delayed(key, f_names, project_info) for key, f_names in tqdm(grouped_f_names.items())))
    logger.info('Merging all interaction counts...')
    interaction_counts = merge_interaction_counts(res)
    res = None  # release memory

    # add interaction data to tweets and write compressed parquet dataframes
    logger.info('Writing parquet files...')
    write_parquet_file_delayed = joblib.delayed(write_parquet_file)
    res = parallel((write_parquet_file_delayed(key, interaction_counts) for key in tqdm(list(grouped_f_names.keys()))))
    num_tweets = sum(res)
    logger.info(f'Collected a total of {num_tweets:,} tweets in {len(grouped_f_names):,} parquet files')

    # removing preliminary files
    preliminary_folder = os.path.join(output_folder, 'preliminary')
    if os.path.isdir(preliminary_folder):
        logger.info('Cleaning up intermediary files...')
        shutil.rmtree(preliminary_folder)

    # write used files
    logger.info('Writing used files...')
    write_used_files(grouped_f_names)

    e_time = time.time()
    logger.info(f'Finished in {(e_time-s_time)/60:.1f} min')

