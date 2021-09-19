"""
The purpose of parse is to generate Parquet files from raw jsonl.gz files

`parse` does three things:
    * Extract relevant fields from tweets
    * Tries to tie tweets to a geographical location (longitude/latitude point) using local-geocode
    * converts raw gzipped raw data into daily Dataframes in parquet format (faster to read)

The parsing takes place in two episodes:
    1. Run extraction and count interactions (retweets, quotes, replies)
    2. Merge interaction counts and write parquet files

The purpose of these two episodes is to minimize memory (never having to fully load all data into memory).

1. Run extraction
-----------------
Input: Raw data (jsonl.gz)
Output: Daily jsonl files (called preliminary files) with the extracted tweets & interaction counts
This function is parallelized using joblib. The interaction counts are held in a thread-safe shared read/write
data structure called multiprocessing Manager dictionary (`manager.dict()`). Additional we keep a dictionary
called `originals` to keep track of duplicates. Duplicates can appear "naturally" in the Twitter stream or when
extracting "subtweets". Subtweets are either quotes statuses or retweeted statuses. We can control whether or not
to extract these subtweets (using the `extract_quotes` and `extract_retweets` args). Note, that when subtweets are
extracted they may have been tweeted several years in the past (before the time of data collection).

2. Merge interactions
--------------------
Input: Preliminary jsonl files & interaction counts
Output: Daily parquet files (written to data/1_parsed/tweets)
This function is parallelized using ray. Ray allows for an arbitrary datastructure (in our case the interaction
counts, which are held in a Pandas DataFrame) to be shared in a read-only way among multiple processes.
The interaction counts are merged with the extracted tweets adding the columns num_retweets, num_quotes, and
num_replies.

Eventually all temporary files (preliminary files) are removed and the parquet files can be read in parallel with
the `utils.helpers.get_parsed_data()` function.

Notes:
* This code was tested on a machine with 250GB of memory. Depending on data size or machine you may want to reduce
the number of parallel workers (using the `--ray_num_cpus` argument).
* After every run a .used_data file is written, keeping track of which raw files were already parsed. This allows
to use the --extend keyword and add new data without re-parsing the whole raw data every time.
* By default the last day (which is an incomplete day of collection) is ignored (see `--omit_last_day` argument)
"""
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
import re

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
        elif os.path.basename(f_name).startswith('crowdbreaks'):
            date_str = re.findall(r'\d{4}\-\d+\-\d+\-\d+\-\d+\-\d+', os.path.basename(f_name))[0]
            day_str = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S').strftime(datefmt)
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
    return grouped_f_names

@ray.remote
def write_parquet_file(f_path_intermediary, interaction_counts, extend=False):
    key = f_path_intermediary.split('/')[-1].split('.jsonl')[0]
    f_out = os.path.join(output_folder, 'tweets', f'parsed_{key}.parquet')
    # Read from json lines
    dtypes = get_dtypes()
    df = pd.read_json(f_path_intermediary, lines=True, dtype=dtypes)
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
        df_counts = None  # Free up memory
        # Append newly collected data
        df = pd.concat([df_existing, df])
        df_existing = None  # Free up memory
        # After concatenation new data is at the bottom and duplicates will be removed from new data
        df = df.drop_duplicates(subset=['id'], keep='first')
        # Sort again
        df.sort_values('created_at', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
    # Write parquet file
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

def run(no_parallel=False, extract_retweets=True, extract_quotes=True, extend=False, omit_last_day=True, ray_num_cpus=None):
    def extract_tweets(day, f_names, keywords):
        gc = Geocode()
        gc.load()
        map_data = load_map_data()
        # Create dirs
        for subfolder in ['tweets', 'preliminary']:
            folder = os.path.join(output_folder, subfolder)
            if not os.path.isdir(folder):
                os.makedirs(folder)
        def write_to_file(obj):
            # extracted_tweet, which is an instance of the ProcessTweet class, is the object represented by the variable obj
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
                    # Some files use single quotation (Python dictionaries), for this we need to use ast.literal_eval
                    tweet = ast.literal_eval(line)
                except:
                    # Sometimes parsing completely fails
                    logger.error('Error parsing line:')
                    logger.error(line)
                    continue
                if not 'id' in tweet:
                    continue  # API logs, error messages, etc.
                tweet_id = tweet['id_str']
                if tweet_id in originals:
                    # Skip duplicates
                    continue
                # Flag tweet ID as "used"
                originals[tweet_id] = True
                # Extract tweet
                pt = ProcessTweet(tweet=tweet, keywords=keywords, map_data=map_data, gc=gc)
                if ((extract_retweets and pt.is_retweet)                              # extract retweets (optional)
                        or (extract_quotes and pt.has_quote and not pt.is_retweet)    # extract quotes if not retweet of a quote (optional)
                        or (not pt.is_retweet and not pt.has_quote)):                 # always extract original tweets which are neither retweets nor quotes
                    extracted_tweet = pt.extract()
                    write_to_file(extracted_tweet)
                # Add interaction counts
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
                        # Extract original status
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
                        # Extract original status
                        originals[pt_retweet.id] = True
                        extracted_tweet = pt_retweet.extract()
                        write_to_file(extracted_tweet)
            f.close()
    # Setup
    s_time = time.time()
    try:
        project_info = get_project_info()
        keywords = project_info['keywords']
    except FileNotFoundError:
        logger.warning('Could not find project info file. Will not compute matching keywords')
        keywords = []

    # Collect file list
    grouped_f_names = generate_file_list(extend=extend, omit_last_day=omit_last_day)

    if len(grouped_f_names) == 0:
        logger.info('No new files found to process. All up-to-date.')
        sys.exit()
    num_files = sum([len(l) for l in grouped_f_names.values()])
    num_days = len(grouped_f_names.keys())
    logger.info(f'About to parse {num_files:,} new files collected within {num_days:,} days...')

    existing_parquet_files = glob.glob(os.path.join(output_folder, 'tweets', '*.parquet'))
    
    if extend and len(existing_parquet_files) == 0:
        raise Exception('The extend argument was passed, but there are no existing files to extend')
    if not extend and len(existing_parquet_files) > 0:
        raise Exception('Found existing parsed files. Either delete pre-existing data manually or pass --extend argument.')

    # Make sure map data is downloaded and geocode is set up
    map_data = load_map_data()
    gc = Geocode()
    gc.load()

    # Set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    logger.info(f'Using {num_cores} CPUs to parse data...')
    parallel = joblib.Parallel(n_jobs=num_cores)

    # Run
    logger.info('Extract tweets...')
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    parallel(extract_tweets_delayed(key, f_names, keywords) for key, f_names in tqdm(grouped_f_names.items()))
    logger.info('Merging all interaction counts...')
    interaction_counts = merge_interaction_counts()
    interaction_counts_fname = dump_interaction_counts(interaction_counts)

    # Store counts as shared memory
    if ray_num_cpus is None and no_parallel:
        ray_num_cpus = 1
    ray.init(num_cpus=ray_num_cpus)
    data_id = ray.put(interaction_counts)

    if extend:
        existing_parquet_keys = [os.path.basename(f_name).split('.parquet')[0][len('parsed_'):] for f_name in existing_parquet_files]
        f_names_intermediary = glob.glob(os.path.join(output_folder, 'preliminary', '*.jsonl'))
        # List of existing Parquet files to extend
        f_names_intermediary_existing = []
        # List of new Parquet files to write
        f_names_intermediary_new = []
        # Fill the above lists
        for f_name in f_names_intermediary:
            if os.path.basename(f_name).split('.jsonl')[0] in existing_parquet_keys:
                f_names_intermediary_existing.append(f_name)
            else:
                f_names_intermediary_new.append(f_name)
        
        # Extend existing Parquet files
        if len(f_names_intermediary_existing) == 0:
            logger.info('No files to extend.')
        else:
            logger.info(f'Extending {len(f_names_intermediary_existing):,} existing parquet files...')
            res = ray.get([write_parquet_file.remote(f_name, data_id, extend=True) for f_name in tqdm(f_names_intermediary_existing)])
            num_existing_tweets = sum(res)
            logger.info(f'... updated existing {num_existing_tweets:,} tweets in {len(f_names_intermediary_existing):,} parquet files')
    else:
        # List of new Parquet files to write
        f_names_intermediary_new = glob.glob(os.path.join(output_folder, 'preliminary', '*.jsonl'))

    # Write new Parquet files
    if len(f_names_intermediary_new) > 0:
        logger.info(f'Writing {len(f_names_intermediary_new):,} new parquet files...')
        res = ray.get([write_parquet_file.remote(f_name, data_id) for f_name in tqdm(f_names_intermediary_new)])
        num_tweets = sum(res)
        logger.info(f'... collected a total of {num_tweets:,} tweets in {len(f_names_intermediary_new):,} parquet files')
    else:
        logger.info('No new Parquet files to write...')
        num_tweets = 0

    # Write used files
    logger.info('Writing used files...')
    if extend:
        # Get full file list
        grouped_f_names = generate_file_list(extend=False, omit_last_day=omit_last_day)
    write_used_files(grouped_f_names)

    # Cleanup
    preliminary_folder = os.path.join(output_folder, 'preliminary')
    if os.path.isdir(preliminary_folder):
        logger.info('Cleaning up intermediary files...')
        shutil.rmtree(preliminary_folder)
    e_time = time.time()
    logger.info(f'Finished in {(e_time-s_time)/3600:.1f} hours')
