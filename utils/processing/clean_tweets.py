import os
import re
import time
import logging
from utils.helpers import get_merged_data, get_f_name, find_file, get_dtypes
from collections import Counter
import spacy
import joblib
import multiprocessing
from tqdm import tqdm
import pandas as pd
import hashlib


def get_text_hashes(chunk):
    hashes = chunk['text'].apply(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest())
    return hashes

def compute_token_count(chunk, lang):
    text = chunk['text'].apply(str)
    nlp = spacy.load(lang)
    # remove user handles and URLs from text
    text = text.apply(lambda t: re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', t))
    text = text.apply(lambda t: re.sub('(\@[^\s]+)', '', t))
    text = text.apply(lambda t: nlp(t, disable=['parser', 'tagger', 'ner']))
    # Count the number of tokens excluding stopwords
    token_count = text.apply(lambda doc: len([token for token in doc if token.is_alpha and not token.is_stop]))
    return token_count

def add_decision_flags(df, token_count_cutoff=3):
    """Add flags which have been pre-defined for later processing steps"""
    # labelling: no retweets, no extracted tweets, no duplicates, min token count
    df['use_for_labelling'] = (~df['is_retweet']) & (~df['extracted_quoted_tweet']) & (~df['is_duplicate']) & (df['token_count'] >= token_count_cutoff)
    # prediction: min token count
    df['use_for_prediction'] = df['token_count'] >= token_count_cutoff
    return df

def run_chunk(chunk, lang='en_core_web_lg'):
    chunk['token_count'] = compute_token_count(chunk, lang)
    chunk['text_hash'] = get_text_hashes(chunk)
    return chunk

def read_chunk(path, chunksize=2**15):
    dtypes = get_dtypes()
    for text_chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize):
        yield text_chunk

def load_in_parallel(dtype, lang, no_parallel=False):
    """Loads data and computes new columns in parallel"""
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cores)
    chunksize = 2**14
    f_name = get_f_name(dtype, None, processing_step='merged')
    f_path = find_file(f_name, subfolder='1_merged')
    with open(f_path, 'r') as f:
        num_it = int(sum([1 for _ in f]) / chunksize) + 1
    input_data = read_chunk(f_path, chunksize=chunksize)
    run_chunk_delayed = joblib.delayed(run_chunk)
    res = parallel((run_chunk_delayed(chunk, lang=lang) for chunk in tqdm(input_data, total=num_it, unit='chunk')))
    return pd.concat(res)

def run(dtypes=['original'], lang='en_core_web_lg', no_parallel=False, verbose=False):
    s_time = time.time()
    logger = logging.getLogger(__name__)
    for dtype in dtypes:
        logger.info('Clean data of type {}...'.format(dtype))
        # compute stuff in parallel
        df = load_in_parallel(dtype, lang, no_parallel=no_parallel)
        # find duplicates
        df['is_duplicate'] = df.duplicated(subset='text_hash', keep='first')
        num_tweets = len(df)
        if verbose:
            non_unique_counts = dict(Counter(df.is_duplicate))[True]
            logger.info('Number of text duplicates: {:,}/{:,} ({:.1f}%)'.format(non_unique_counts, num_tweets, 100*non_unique_counts/num_tweets))
            mean_token_count = df.token_count.mean()
            median_token_count = df.token_count.median()
            logger.info('Token counts: Mean: {:.2f}, Median: {:.2f}'.format(mean_token_count, median_token_count))
        # decision flags
        df = add_decision_flags(df, token_count_cutoff=3)
        if verbose:
            labelling_counts = dict(Counter(df.use_for_labelling))[True]
            prediction_counts = dict(Counter(df.use_for_prediction))[True]
            logger.info('Marked to be used for annotation: {:,}/{:,} ({:.1f}%)'.format(labelling_counts, num_tweets, 100*labelling_counts/num_tweets))
        # write output file
        f_name = 'cleaned_{}.csv'.format(dtype)
        out_path = os.path.join('data', '2_cleaned', f_name)
        logger.info('Writing file {}...'.format(out_path))
        df.set_index('created_at', inplace=True)
        df.to_csv(out_path, encoding='utf8')
    e_time = time.time()
    logger.info('... done after {:.1f} min'.format((e_time - s_time)/60.0))
