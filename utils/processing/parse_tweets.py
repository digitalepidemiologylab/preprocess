import pandas as pd
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
from local_geocode.geocode.geocode import Geocode
import shutil
import pickle
import gzip


logger = logging.getLogger(__name__)
output_folder = os.path.join('data', '1_parsed')
input_folder = os.path.join('data', '0_raw')


manager = multiprocessing.Manager()
# shared between all processes
originals = manager.dict()
retweet_counts = manager.dict()
quote_counts = manager.dict()
replies_counts = manager.dict()


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
    f_names = []
    for file_type in ['jsonl', 'gz']:
        globbed = glob.glob(os.path.join(input_folder, '**', f'*.{file_type}'))
        f_names.extend(globbed)
    grouped_f_names = defaultdict(list)
    for f_name in f_names:
        date_str = f_name.split('-')[1]
        day_str = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime('%Y-%m-%d')
        grouped_f_names[day_str].append(f_name)
    return grouped_f_names

def write_parquet_file(f_path_intermediary, interaction_counts):
    # read from json lines
    dtypes = get_dtypes()
    key = f_path_intermediary.split('/')[-1].split('.jsonl')[0]
    df = pd.read_json(f_path_intermediary, lines=True, dtype=dtypes)
    if len(df) > 0:
        # drop duplicates
        df.drop_duplicates(subset=['id'], inplace=True)
        # read_json converts null to stringified 'None', convert manually
        for col in [c for c, v in dtypes.items() if v == str]:
            df.loc[df[col] == 'None', col] = None
        # merge with interaction counts
        df = df.merge(interaction_counts, on='id', how='left')
        for col in ['num_replies', 'num_quotes', 'num_retweets']:
            df[col] = df[col].fillna(0).astype(int)
        # convert columns to datetime
        for datetime_col in ['created_at', 'user.created_at']:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        # convert to categorical types
        for col in ['country_code', 'region', 'subregion', 'geo_type', 'lang']:
            df[col] = df[col].astype('category')
        # sort by created_at
        df.sort_values('created_at', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        # write parquet file
        f_out = os.path.join(output_folder, 'tweets', f'parsed_{key}.parquet')
        df.to_parquet(f_out)
    return len(df)

def merge_interaction_counts():
    interaction_counts = pd.DataFrame({
        'num_quotes': pd.Series(dict(quote_counts)),
        'num_replies': pd.Series(dict(replies_counts)),
        'num_retweets': pd.Series(dict(retweet_counts))})
    interaction_counts.index.name = 'id'
    for col in ['num_quotes', 'num_replies', 'num_retweets']:
        interaction_counts[col] = interaction_counts[col].fillna(0).astype(int)
    interaction_counts.reset_index(inplace=True)
    return interaction_counts

def dump_interaction_counts(interaction_counts):
    """Cache interaction counts in case something goes wrong"""
    now = datetime.now().isoformat()
    f_name = os.path.join('/', 'tmp', f'interaction_counts_{now}.pkl')
    logger.info(f'Writing interaction counts to temporary file {f_name}...')
    with open(f_name, 'wb') as f:
        pickle.dump(dict(interaction_counts), f)
    return f_name

def run(lang='en_core_web_sm', no_parallel=False):
    def extract_tweets(day, f_names, project_info):
        gc = Geocode()
        gc.init()
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
            else:
                f = open(f_name, 'r')
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
                tweet_id = tweet['id_str']
                if tweet_id in originals:
                    # skip duplicates
                    continue
                # create new entry in interaction counts
                originals[tweet_id] = True
                # extract tweet
                pt = ProcessTweet(tweet=tweet, project_info=project_info, map_data=map_data, gc=gc)
                extracted_tweet = pt.extract()
                write_to_file(extracted_tweet)
                # add interaction counts
                if pt.is_reply:
                    if pt.replied_status_id in replies_counts:
                        replies_counts[pt.replied_status_id] += 1
                    else:
                        replies_counts[pt.replied_status_id] = 1
                if pt.has_quote:
                    pt = ProcessTweet(tweet=tweet['quoted_status'], project_info=project_info, map_data=map_data, gc=gc)
                    if pt.id in quote_counts:
                        quote_counts[pt.id] += 1
                    else:
                        quote_counts[pt.id] = 1
                    if not pt.id in originals:
                        # extract original status
                        originals[pt.id] = True
                        extracted_tweet = pt.extract()
                        write_to_file(extracted_tweet)
                if pt.is_retweet:
                    pt = ProcessTweet(tweet=tweet['retweeted_status'], project_info=project_info, map_data=map_data, gc=gc)
                    if pt.id in retweet_counts:
                        retweet_counts[pt.id] += 1
                    else:
                        retweet_counts[pt.id] = 1
                    if pt.id not in originals:
                        # extract original status
                        originals[pt.id] = True
                        extracted_tweet = pt.extract()
                        write_to_file(extracted_tweet)
            f.close()
    # setup
    s_time = time.time()
    project_info = get_project_info()

    # collect file list
    grouped_f_names = generate_file_list()
    # make sure map data is downloaded and geocode is set up
    map_data = load_map_data()
    gc = Geocode()
    gc.prepare()

    # set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    logger.info(f'Using {num_cores} CPUs to parse data...')
    parallel = joblib.Parallel(n_jobs=num_cores)

    # run
    logger.info('Extract tweets...')
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    parallel(extract_tweets_delayed(key, f_names, project_info) for key, f_names in tqdm(grouped_f_names.items()))
    logger.info('Merging all interaction counts...')
    interaction_counts = merge_interaction_counts()
    interaction_counts_fname = dump_interaction_counts(interaction_counts)

    # add interaction data to tweets and write compressed parquet dataframes
    logger.info('Writing parquet files...')
    write_parquet_file_delayed = joblib.delayed(write_parquet_file)
    f_names_intermediary = glob.glob(os.path.join(output_folder, 'preliminary', '*.jsonl'))
    res = parallel((write_parquet_file_delayed(key, interaction_counts) for key in tqdm(f_names_intermediary)))
    num_tweets = sum(res)
    logger.info(f'Collected a total of {num_tweets:,} tweets in {len(f_names_intermediary):,} parquet files')

    # write used files
    logger.info('Writing used files...')
    write_used_files(grouped_f_names)

    # cleanup
    preliminary_folder = os.path.join(output_folder, 'preliminary')
    if os.path.isdir(preliminary_folder):
        logger.info('Cleaning up intermediary files...')
        shutil.rmtree(preliminary_folder)
    if os.path.isfile(interaction_counts_fname):
        logger.info('Cleaning up counts file...')
        os.remove(interaction_counts_fname)
    e_time = time.time()
    logger.info(f'Finished in {(e_time-s_time)/60:.1f} min')
