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
import resource

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

def generate_file_list(extend=False, omit_last_day=True):
    """generates dictionary of files per day"""
    f_names = []
    for file_type in ['jsonl', 'gz', 'bz2']:
        globbed = Path(input_folder).rglob(f'*.{file_type}')
        f_names.extend([str(g) for g in globbed])
    grouped_f_names = defaultdict(list)
    datefmt = '%Y-%m-%d'
    if extend:
        used_files = read_used_files()
    for f_name in f_names:
        if os.path.basename(f_name).startswith('tweets'):
            date_str = f_name.split('-')[1]
            day_str = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime(datefmt)
        else:
             day_str = '-'.join(f_name.split('/')[-5:-2])
        if extend:
            if day_str in used_files and f_name in used_files[day_str]:
                continue
        grouped_f_names[day_str].append(f_name)
    if omit_last_day:
        # Remove last incomplete day (this makes extending data easier)
        if len(grouped_f_names) > 0:
            max_date = max([datetime.strptime(key, datefmt) for key in grouped_f_names.keys()])
            max_date_key = max_date.strftime(datefmt)
            grouped_f_names.pop(max_date_key)
            if len(grouped_f_names) == 0 and not extend:
                logger.warning('Found data from only a single day. Use --do_not_omit_last_day option to parse data')
    # other_dates_omitted = sorted([datetime.strptime(key, datefmt) for key in grouped_f_names.keys()])[-499:]
    return grouped_f_names

@ray.remote
def write_parquet_file(f_path_intermediary, interaction_counts, extend=False):
    key = f_path_intermediary.split('/')[-1].split('.jsonl')[0]
    f_out = os.path.join(output_folder, 'tweets', f'parsed_{key}.parquet')
    # Read from JSON lines
    dtypes = get_dtypes()
    df = pd.read_json(os.path.join(output_folder, 'preliminary', f_path_intermediary), lines=True, dtype=dtypes)
    if len(df) == 0:
        return 0
    # Drop duplicates
    df.drop_duplicates(subset=['id'], inplace=True)
    # read_json converts null to stringified 'None', convert manually
    for col in [c for c, v in dtypes.items() if v == str]:
        df.loc[df[col] == 'None', col] = None
    # Sanity check, verify uniqueness of IDs
    df = df.drop_duplicates(subset=['id'])
    # Merge with interaction counts
    if len(interaction_counts) > 0:
        # Subsetting interaction counts to save memory during merge
        interaction_counts_new = interaction_counts[interaction_counts.id.isin(df.id.unique())]
        df = df.merge(interaction_counts_new, on='id', how='left')
        for col in ['num_replies', 'num_quotes', 'num_retweets']:
            df[col] = df[col].fillna(0).astype(int)
    else:
        # Set default values
        for col in ['num_replies', 'num_quotes', 'num_retweets']:
            df[col] = 0
    # Convert columns to datetime
    for datetime_col in ['created_at', 'user.created_at']:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    # Convert to categorical types
    for col in ['country_code', 'region', 'subregion', 'location_type', 'geo_type', 'lang']:
        df[col] = df[col].astype('category')
    # Sort by created_at
    df.sort_values('created_at', inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)
    if extend:
        # Read pre-existing parquet files
        df_existing = pd.read_parquet(f_out)
        # Sum up previous and new interaction counts
        interaction_counts_new = interaction_counts[interaction_counts.id.isin(df_existing.id.unique())]
        merge_cols = ['id', 'num_replies', 'num_quotes', 'num_retweets']
        df_counts = pd.merge(df_existing[merge_cols], interaction_counts_new, on='id', how='left').fillna(0)
        for col in merge_cols[1:]:
            sum_cols = [f'{col}_x', f'{col}_y']
            df_counts[col] = df_counts[sum_cols].sum(axis=1)
            df_counts[col] = df_counts[col].astype(int)
            df_existing[col] = df_counts[col].values
        df_counts = None  # free up memory
        # append newly collected data
        df = pd.concat([df_existing, df])
        df_existing = None  # free up memory
        # after concatenation new data is at the bottom and duplicates will be removed from new data
        df = df.drop_duplicates(subset=['id'], keep='first')
        # sort again
        df.sort_values('created_at', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
    # write parquet file
    df.to_parquet(f_out)
    return len(df)

def main():
    # Setup
    s_time = time.time()

    no_parallel = False
    extend = False

    # Set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    # logger.info(f'Using {num_cores} CPUs to parse data...')

    parallel = joblib.Parallel(n_jobs=num_cores)
    
    max_rec = 7000000000
    # Trying to pickle a highly recursive data structure may exceed 
    # the maximum recursion depth, a RuntimeError will be raised in this case. 
    # You can carefully raise this limit with sys.setrecursionlimit()
    sys.setrecursionlimit(int(max_rec/100))
    # Increase stack size with resource.setrlimit in order to prevent segfault 
    resource.setrlimit(resource.RLIMIT_STACK, [max_rec, -1])
    
    interaction_counts_f_name = os.path.join('/', 'tmp', 'interaction_counts_2021-09-07T12:14:00.302315.pkl')
    file_to_read = open(interaction_counts_f_name, 'rb')
    interaction_counts = pickle.load(file_to_read)
    # Interaction_counts is a dictionary of Pandas Series; turn it into a DataFrame
    interaction_counts = pd.DataFrame(interaction_counts)
    
    # Store counts as shared memory
    # if ray_num_cpus is None and no_parallel:
        # ray_num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    ray_num_cpus = 5
    ray.init(num_cpus=ray_num_cpus)
    data_id = ray.put(interaction_counts)
    # Omit last 500 days: the last day is already omitted in 1_parsed/preliminary, and the 499 remaining days (500 last days = {final_day-499, final_day-498, ..., final_day-1, final_day}, where final_day is already omitted in 1_parsed/preliminary) correspond to the last 499 days in 1_parsed/preliminary ({final_day-499, final_day-498, ..., final_day-2, final_day-1})
    # f_names_intermediary_new = sorted(os.listdir(os.path.join(output_folder, 'preliminary')))[:-499]
    f_names_intermediary_new = sorted(os.listdir(os.path.join(output_folder, 'preliminary')))[-181:]

    # Write new parquet files
    if len(f_names_intermediary_new) > 0:
        logger.info(f'Writing {len(f_names_intermediary_new):,} new parquet files...')
        res = ray.get([write_parquet_file.remote(f_name, data_id) for f_name in tqdm(f_names_intermediary_new)])
        num_tweets = sum(res)
        logger.info(f'... collected a total of {num_tweets:,} tweets in {len(f_names_intermediary_new):,} parquet files')
    else:
        logger.info('No new parquet files to write...')
        num_tweets = 0
    
    omit_last_day = True
    # Write used files
    logger.info('Writing used files...')
    if extend:
        # Get full file list
        grouped_f_names = generate_file_list(extend=False, omit_last_day=omit_last_day)
    write_used_files(grouped_f_names)

    # Cleanup
    # preliminary_folder = os.path.join(output_folder, 'preliminary')
    # if os.path.isdir(preliminary_folder):
    #     logger.info('Cleaning up intermediary files...')
    #     shutil.rmtree(preliminary_folder)
    e_time = time.time()
    logger.info(f'Finished in {(e_time-s_time)/3600:.1f} hours')
    

if __name__ == '__main__':
    main()