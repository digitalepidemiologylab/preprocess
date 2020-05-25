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


logger = logging.getLogger(__name__)


def get_used_files(output_path):
    f_path = os.path.join(output_path, f'.used_data')
    if not os.path.isfile(f_path):
        return {}
    with open(f_path, 'r') as f:
        used_files = json.load(f)
    return used_files

def dump_used_files(data_files, output_path):
    f_name_used = os.path.join(output_path, f'.used_data')
    with open(f_name_used, 'w') as f:
        json.dump(data_files, f, indent=4)

def generate_file_list():
    """generates dictionary of files per day"""
    output_path = os.path.join('data', '0_raw', 'streaming', '*json?')
    f_names = glob.glob(output_path)
    grouped_f_names = defaultdict(list)
    for f_name in f_names:
        date_str = f_name.split('-')[1]
        day_str = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime('%Y-%m-%d')
        grouped_f_names[day_str].append(f_name)
    return grouped_f_names

def extract_tweets(day, f_names, project_info):
    count = 0
    gc = Geocode()
    gc.init()
    map_data = load_map_data()
    tweet_interaction_counts = defaultdict(lambda: {'num_quotes': 0, 'num_retweets': 0, 'num_replies': 0})
    f_out_folder = os.path.join('data', '1_parsed', 'preliminary')
    f_out = os.path.join(f_out_folder, f'{day}.jsonl')
    if not os.path.isdir(f_out_folder):
        os.makedirs(f_out_folder)
    if os.path.isfile(f_out):
        os.remove(f_out)
    def write_to_file(obj):
        with open(f_out, 'a') as f:
            f.write(json.dumps(obj) + '\n')
    for f_name in f_names:
        with open(f_name, 'r') as f:
            num_lines = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(f, total=num_lines):
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
                pt = ProcessTweet(tweet=tweet, project_info=project_info, map_data=map_data, gc=gc)
                if pt.is_retweet:
                    tweet_interaction_counts[pt.retweeted_status_id]['num_retweets'] += 1
                if pt.has_quoted_status:
                    tweet_interaction_counts[pt.quoted_status_id]['num_quotes'] += 1
                if pt.is_reply:
                    tweet_interaction_counts[pt.replied_status_id]['num_replies'] += 1
                # extract tweet/retweet/replies
                extracted_tweet = pt.extract()
                text_hash =  pt.get_text_hash()
                write_to_file(extracted_tweet)
                count +=1
                if pt.has_quoted_status:
                    # extract quoted status
                    pt = ProcessTweet(tweet=tweet['quoted_status'], project_info=project_info, map_data=map_data, gc=gc)
                    extracted_tweet = pt.extract()
                    write_to_file(extracted_tweet)
                    count +=1
    return tweet_interaction_counts, f_out

def write_parquet_file(output_file, interaction_counts):
    df = []
    # add interaction counts for every tweet
    with open(output_file, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweet = {**tweet, **interaction_counts[tweet['id']]}
            df.append(tweet)
    filename = os.path.basename(output_file).split('.jsonl')[0]
    f_out_folder = os.path.join('data', '1_parsed')
    f_out = os.path.join(f_out_folder, f'parsed_{filename}.parquet')
    # drop duplicates
    df = pd.DataFrame(df)
    df.drop_duplicates(subset=['id'], inplace=True)
    # write parquet file
    df.to_parquet(f_out)
    # delete old file
    os.remove(output_file)
    return len(df), f_out

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
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    res = parallel((extract_tweets_delayed(key, f_names, project_info) for key, f_names in tqdm(grouped_f_names.items())))
    logger.info('Merging all interaction counts...')
    interaction_counts = [dict(r[0]) for r in res]
    interaction_counts = merge_interaction_counts(interaction_counts)

    # add interaction data to tweets and write compressed parquet dataframes
    logger.info('Writing parquet files...')
    output_files = [r[1] for r in res]
    write_parquet_file_delayed = joblib.delayed(write_parquet_file)
    res = parallel((write_parquet_file_delayed(output_file, interaction_counts) for output_file in tqdm(output_files)))
    num_tweets = sum(r[0] for r in res)
    output_files = [r[1] for r in res]

    # remove duplicates
    logger.info('Removing duplicates...')
    duplicates = set()
    removed = 0
    num_tweets = 0
    for f_out in tqdm(output_files):
        df = pd.read_parquet(f_out)
        num_before = len(df)
        df = df[~df.id.isin(duplicates)]
        num_after = len(df)
        removed += num_before - num_after
        # add ids to duplicates
        duplicates.update(df.id.tolist())
        # overwrite
        df.to_parquet(f_out)
        num_tweets += len(df)
    logger.info(f'...removed {removed:,} duplicates')
    e_time = time.time()
    logger.info(f'Collected a total of {num_tweets:,} tweets in {len(output_files):,} parquet files')
    logger.info(f'Finished in {(e_time-s_time)/60:.1f} min')
