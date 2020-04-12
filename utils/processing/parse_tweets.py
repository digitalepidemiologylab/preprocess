import pandas as pd
import os
from copy import copy
from collections import defaultdict
from utils.processing.encrypt import Encrypt
import glob
import ast
import time
import json
from tqdm import tqdm
from munch import DefaultMunch
import multiprocessing
import joblib
import logging
import random
from utils.helpers import get_cache_path, get_dtypes, get_project_info, cache_folder, get_limited_cols
from utils.process_tweet import ProcessTweet
import shutil
import pickle


logger = logging.getLogger(__name__)

def get_cache_location_from_fname(f_name, dtype):
    f_name_base = os.path.basename(f_name)
    f_path_cache = get_cache_path(f'{f_name_base}.{dtype}.pkl', subfolder='parse_tweets')
    return f_path_cache

def process_file(f_name, config):
    all_data = {output_type: [] for output_type in config.output_types}
    count = 0
    if config.output_types.encrypted:
        encrypt = Encrypt()
    with open(f_name, 'r') as f:
        num_lines = sum(1 for line in f)
        f.seek(0)
        for line in f:
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
            pt = ProcessTweet(tweet=tweet)
            tweet_obj = {
                    **pt.get_fields(config.keep_fields),
                    **pt.get_user_fields(config.keep_fields_user),
                    **pt.get_entities_fields(config.keep_fields_entities),
                    **pt.get_place_fields(config.keep_fields_place),
                    **pt.get_coordinates_field(),
                    **pt.get_retweet_info(config.keep_fields_retweeted_status),
                    **pt.get_quoted_status_info(config.keep_fields_quoted_status),
                    **pt.get_media_info(),
                    'extracted_quoted_tweet': False,
                    'contains_keywords': pt.contains_keywords(config.keywords),
                    'token_count': pt.get_token_count(),
                    'text_hash': pt.get_text_hash()
                    }
            if config.output_types.original:
                all_data['original'].append(tweet_obj)
            if config.output_types.anonymized:
                tweet_obj_anonymized = pt.anonymize(tweet_obj)
                all_data['anonymized'].append(tweet_obj_anonymized)
            if config.output_types.encrypted:
                if not config.output_types.anonymized:
                    tweet_obj_anonymized = pt.anonymize(tweet_obj)
                all_data['encrypted'].append(encrypt.encode_tweet(copy(tweet_obj_anonymized), config.encrypt_fields))
            count +=1
            # append quoted tweet as normal tweet
            if tweet_obj['has_quoted_status']:
                pt = ProcessTweet(tweet=tweet['quoted_status'])
                tweet_obj = {
                        **pt.get_fields(config.keep_fields),
                        **pt.get_user_fields(config.keep_fields_user),
                        **pt.get_entities_fields(config.keep_fields_entities),
                        **pt.get_place_fields(config.keep_fields_place),
                        **pt.get_coordinates_field(),
                        **pt.get_retweet_info(config.keep_fields_retweeted_status),
                        **pt.get_quoted_status_info(config.keep_fields_quoted_status),
                        **pt.get_media_info(),
                        'extracted_quoted_tweet': True,
                        'contains_keywords': pt.contains_keywords(config.keywords),
                        'token_count': pt.get_token_count(),
                        'text_hash': pt.get_text_hash()
                        }
                if config.output_types.original:
                    all_data['original'].append(tweet_obj)
                if config.output_types.anonymized:
                    tweet_obj_anonymized = pt.anonymize(tweet_obj)
                    all_data['anonymized'].append(tweet_obj_anonymized)
                if config.output_types.encrypted:
                    if not config.output_types.anonymized:
                        tweet_obj_anonymized = pt.anonymize(tweet_obj)
                    all_data['encrypted'].append(encrypt.encode_tweet(copy(tweet_obj_anonymized), config.encrypt_fields))
                count +=1
    # writing cache file
    for dtype, df in all_data.items():
        f_path_cache = get_cache_location_from_fname(f_name, dtype)
        with open(f_path_cache, 'wb') as f:
            pickle.dump(all_data[dtype], f)

def get_used_files(output_path):
    f_name_used = os.path.join(output_path, f'.used_data')
    if os.path.isfile(f_name_used):
        with open(f_name_used, 'r') as f:
            used_files = list(l.strip() for l in f)
    else:
        used_files = []
    return used_files

def dump_used_files(data_files, output_path):
    f_name_used = os.path.join(output_path, f'.used_data')
    with open(f_name_used, 'w') as f:
        for data_file in data_files:
            f.write(f'{data_file}\n')
    
def write_csv(df, f_name):
    df.to_csv(f_name, index=False, header=False)

def write_csv_in_parallel(df, f_name, no_parallel):
    batch_size = int(1e6)
    num_rows = len(df)
    shutil.rmtree(cache_folder(subfolder='partial-csv-write'))
    indices = range(0, num_rows, batch_size)
    num_jobs = len(indices)
    if no_parallel:
        num_parallel_jobs = 1
    else:
        # Optimized for 250 GB of RAM
        num_cpus = min(max(multiprocessing.cpu_count() - 1, 1), num_jobs)  # make sure not to use more cores than jobs
        num_parallel_jobs = min(max(int(num_jobs/30), 1), num_cpus)  # reduce by a factor so everything fits into memory
    logger.info(f'Writing {num_jobs} CSVs using {num_parallel_jobs} parallel jobs...')
    cache_names = [get_cache_path(f'partial_csv_{i}.csv', subfolder='partial-csv-write') for i in range(num_jobs)]
    parallel = joblib.Parallel(n_jobs=num_parallel_jobs)
    write_csv_delayed = joblib.delayed(write_csv)
    dfs = (df[s:(s+batch_size)] for s in indices)
    parallel((write_csv_delayed(_df, cache_names[i]) for i, _df in tqdm(enumerate(dfs), total=num_jobs)))
    del dfs
    logger.info('Merging csvs...')
    # write header
    df = pd.DataFrame(columns=df.columns)
    df.to_csv(f_name, index=False)
    # concatenate file contents
    with open(f_name, 'a') as f:
        for cache_name in tqdm(cache_names):
            shutil.copyfileobj(open(cache_name, 'r'), f)
    logger.info('Cleanup...')
    shutil.rmtree(cache_folder(subfolder='partial-csv-write'))

def run(dtypes=['original'], formats=[], lang='en_core_web_sm', no_parallel=False, overwrite=False, extend=False, limited_cols=False, num=None):
    # setup
    s_time = time.time()
    # build config
    config = DefaultMunch(None)
    config.input_data_path = os.path.join('data', '0_raw')
    config.output_data_path = os.path.join('data', '1_parsed')
    # fields to keep
    config.keep_fields = ['id', 'created_at', 'text', 'in_reply_to_status_id', 'in_reply_to_user_id', 'reply_count', 'retweet_count', 'favorite_count', 'lang']
    config.keep_fields_user = ['id', 'screen_name', 'name', 'location', 'followers_count', 'friends_count']
    config.keep_fields_entities = ['hashtags', 'user_mentions']
    config.keep_fields_place = ['bounding_box', 'full_name', 'country_code', 'place_type']
    config.keep_fields_quoted_status = ['quoted_status.id', 'quoted_status.text', 'quoted_status.user.id', 'quoted_status.user.followers_count','quoted_status.in_reply_to_status_id', 'quoted_status.retweet_count', 'quoted_status.favorite_count']
    config.keep_fields_retweeted_status = ['retweeted_status.id', 'retweeted_status.user.id', 'retweeted_status.user.followers_count', 'retweeted_status.in_reply_to_status_id', 'retweeted_status.retweet_count', 'retweeted_status.favorite_count']
    config.encrypt_fields = DefaultMunch.fromDict({'entities.user_mentions': list, 'id': str, 'in_reply_to_status_id': str, 
                      'in_reply_to_user_id': str, 'user.id': str, 'user.screen_name': str, 'user.name': str,
                      'entities.user_mentions': list, 'retweeted_status.id': str, 'retweeted_status.user.id': str, 'retweeted_status.in_reply_to_status_id': str,
                      'quoted_status.id': str, 'quoted_status.user.id': str, 'quoted_status.in_reply_to_status_id': str}, None)
    project_info = get_project_info()
    config.keywords = project_info['keywords']
    config.lang = lang
    # params
    config.output_types = DefaultMunch.fromDict({dtype: True for dtype in dtypes}, False)
    # make sure encryption key is set
    if config.output_types.encrypted:
        Encrypt.verify_encryption_key()
    # run
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    # check for overwrite
    if not overwrite and not extend:
        for t in config.output_types:
            for fmt in formats:
                f_name = os.path.join(config.output_data_path, f'parsed_{t}.{fmt}')
                if os.path.isfile(f_name):
                    raise Exception(f'File {f_name} already exists! Provide --overwrite flag or --extend flag (or remove file)')
    # check for extend
    if extend:
        for t in config.output_types:
            f_name = os.path.join(config.output_data_path, f'parsed_{t}.csv')
            if not os.path.isfile(f_name):
                raise Exception(f'No file {f_name} found to extend')
    parallel = joblib.Parallel(n_jobs=num_cores)
    process_file_delayed = joblib.delayed(process_file)
    all_data_files = sorted([f for f in glob.glob(os.path.join(config.input_data_path, '**', '*.json*'), recursive=True) if os.path.isfile(f)])
    # extend
    if overwrite:
        files_to_cache = all_data_files
    else:
        # check for old file to extend
        cached_files = []
        for f_name in all_data_files:
            missing_type = False
            for t in config.output_types:
                f_path_cache = get_cache_location_from_fname(f_name, t)
                if not os.path.isfile(f_path_cache):
                    missing_type = True
            if not missing_type:
                cached_files.append(f_name)
        files_to_cache = list(set(all_data_files) - set(cached_files))
    # create batches of file names and process in parallel
    if len(files_to_cache) > 0:
        logger.info('Running jobs...')
        random.shuffle(files_to_cache)
        parallel((process_file_delayed(data_file, config) for data_file in tqdm(files_to_cache)))
    else:
        logger.info('All cache files up-to-date.')

    def write_df(_df, dtype='anonymized', fmt='pkl'):
        f_name = os.path.join(config.output_data_path, f'parsed_{t}.{fmt}')
        logger.info(f'Writing {f_name}...')
        if fmt == 'pkl':
            _df.to_pickle(f_name)
        elif fmt == 'h5':
            _df.to_hdf(f_name, key='df')
        elif fmt == 'csv':
            write_csv_in_parallel(df, f_name, no_parallel)
        elif fmt == 'json':
            _df.to_json(f_name)
        else:
            raise ValueError(f'Format {fmt} is not supported')

    def add_decision_flags(df, token_count_cutoff=3):
        """Add flags which have been pre-defined for later processing steps"""
        # labelling: no retweets, no extracted tweets, no duplicates, min token count
        df['use_for_labelling'] = (~df['is_retweet']) & (~df['extracted_quoted_tweet']) & (~df['is_duplicate']) & (df['token_count'] >= token_count_cutoff)
        # prediction: min token count
        df['use_for_prediction'] = df['token_count'] >= token_count_cutoff
        return df

    def report_counts(stats):
        total_size = set([stats[t]['final_counts'] for t, vals in stats.items()])
        if len(total_size) > 1:
            logger.warn('Collected datatypes are of different sizes. This should normally not happen.')
        for t, vals in stats.items():
            logger.info(f'Stats for dtype {t}:')
            for k, v in vals.items():
                logger.info(f'- {k}: {v:,}')

    # merge
    stats = defaultdict(dict)
    limited_columns = get_limited_cols()
    if extend:
        used_files = get_used_files(config.output_data_path)
        data_files = list(set(all_data_files) - set(used_files))
        if len(data_files) == 0:
            logger.info(f'Nothing to extend. Everything up-to-date.')
            return
    else:
        data_files = all_data_files
    if num is not None:
        data_files = data_files[:num]

    for t in config.output_types:
        if config.output_types[t]:
            logger.info(f'Processing data type {t}')
            logger.info(f'Merging {len(data_files):,} cache files...')
            f_names = [get_cache_location_from_fname(f_name, t) for f_name in data_files]
            df = []
            for i, f_name in tqdm(enumerate(f_names), total=len(f_names)):
                with open(f_name, 'rb') as f:
                    try:
                        d = pickle.load(f)
                    except EOFError:
                        continue
                if limited_cols:
                    d = [{k: _d.get(k) for k in limited_columns if k in _d} for _d in d]
                df.extend(d)
            df = pd.DataFrame(df)
            df['created_at'] = pd.to_datetime(df['created_at'])
            if extend:
                f_name = os.path.join(config.output_data_path, f'parsed_{t}.csv')
                if os.path.isfile(f_name):
                    logger.info(f'Merging with pre-existing data from {f_name}...')
                    usecols = None
                    if limited_cols:
                        usecols = limited_columns
                    dtypes = get_dtypes(usecols=usecols)
                    df = pd.concat([df, pd.read_csv(f_name, dtype=dtypes, parse_dates=['created_at'], usecols=usecols)], sort=False)
            if len(df) == 0:
                logger.warning(f'No data was collected for data type {t}')
                continue
            logger.info(f'Collected {len(df):,} rows of data')
            logger.info('Sorting by column created_at...')
            df.sort_values('created_at', inplace=True)
            stats[t]['original_counts'] = len(df)
            # Drop dulicates by ID
            df.drop_duplicates(subset='id', keep='first', inplace=True)
            stats[t]['final_counts'] = len(df)
            # find text duplicates
            logger.info('Find text duplicates...')
            df['is_duplicate'] = df.duplicated(subset='text_hash', keep='first')
            stats[t]['num_text_duplicates'] = len(df[df['is_duplicate']])
            logger.info('Add decision flags...')
            df = add_decision_flags(df)
            stats[t]['num_use_for_labelling'] = len(df[df['use_for_labelling']])
            stats[t]['num_use_for_prediction'] = len(df[df['use_for_prediction']])
            logger.info('Writing data of type {} ...'.format(t))
            for fmt in formats:
                write_df(df, dtype=t, fmt=fmt)
    e_time = time.time()
    report_counts(stats)
    logger.info('Finished after {:.1f} min'.format((e_time - s_time)/60.0))
    dump_used_files(all_data_files, config.output_data_path)
