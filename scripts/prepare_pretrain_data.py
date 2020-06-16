import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_parsed_data, find_folder
from utils.misc import ArgParseDefault
import logging
from tqdm import tqdm, trange
import re
import joblib
import multiprocessing
import datetime
from utils.process_tweet import ProcessTweet


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
user_handle_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

def main(args):
    """
    This script creates a new files in preprocess/data/other/pretrain with tweets which should be used for pretraining language models.
    It excludes training data and duplicates.
    """
    # load data
    logger.info('Reading data...')
    usecols = ['id', 'text', 'lang', 'token_count', 'is_retweet', 'contains_keywords']
    df = get_parsed_data(usecols=usecols, num_files=args.num_files)
    logger.info(f'...loaded a total of {len(df):,} tweets')

    # Filter retweets
    if 'retweets' in args.filters:
        logger.info(f'Filter retweets...')
        num_before = len(df)
        df = df[~df.is_retweet]
        num_after = len(df)
        logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    # Filtering by keyword
    if 'contains_keywords' in args.filters:
        logger.info(f'Filter contains_keywords...')
        num_before = len(df)
        df = df[df.contains_keywords]
        num_after = len(df)
        logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    # filter lang
    if args.lang is not None:
        logger.info(f'Filter lang {args.lang}...')
        num_before = len(df)
        df = df[df.lang == args.lang]
        num_after = len(df)
        logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    # filter min tokens
    if args.min_tokens > 0:
        logger.info(f'Filter has >={args.min_tokens} tokens...')
        num_before = len(df)
        df = df[df.token_count >= args.min_tokens]
        num_after = len(df)
        logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    # generate text column to filter for duplicates
    logger.info('Remove duplicates...')
    num_before = len(df)
    df.loc[:, 'text_cleared'] = df.text.apply(generate_text_cleared)
    df = df.drop_duplicates(subset=['text_cleared'])
    num_after = len(df)
    logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    # shuffle
    logger.info('Shuffle...')
    df = df.sample(frac=1)

    # write output file
    num_lines = len(df)
    logger.info(f'Collected total of {num_lines:,} examples')
    num_train = max(int(0.8*num_lines), num_lines - int(2e5))
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%s')
    for (_s, _e), _type in zip([(None, num_train), (num_train, None)], ['train', 'dev']):
        _df = df[_s:_e]
        logger.info(f'Writing {len(_df):,} examples for {_type} data...')
        output_folder = os.path.join(find_folder('other'), 'pretrain', f'run_{ts}', _type)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        if args.no_parallel:
            num_cpus = 1
        else:
            num_cpus = max(multiprocessing.cpu_count() - 1, 1)
        parallel = joblib.Parallel(n_jobs=num_cpus)
        write_output_file_delayed = joblib.delayed(write_output_file)
        res = parallel((write_output_file_delayed(
            _df.iloc[i:(i+args.max_examples_per_file)], 
            os.path.join(output_folder, f'pretrain_{_type}_{j:03}.txt')
            ) for j, i in enumerate(trange(0, len(_df), args.max_examples_per_file))))
        logger.info(f'Successfully wrote {len(res):,} file(s) to folder {output_folder}')

def write_output_file(df, f_path):
    with open(f_path, 'w') as f:
        num_lines = len(df)
        for i, text in tqdm(df.text.iteritems(), total=num_lines):
            f.write(text + '\n')
    return 1

def generate_text_cleared(text):
    # anonymize text
    text = ProcessTweet.anonymize_text(text)
    # lowercase everything
    text = text.lower()
    # Remove RT @<user>, @<user> and <url>
    text = re.sub(r'rt @user:|@user|<url>', '', text)
    # Remove induced whitespaces
    text = ' '.join(text.split())
    return text

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--filters', choices=['retweets', 'contains_keywords'], default=['retweets'], help="Apply filters")
    parser.add_argument('--lang', default='en', help="Filter language")
    parser.add_argument('--min_tokens', default=3, help="Min num tokens")
    parser.add_argument('--max_examples_per_file', default=int(1e6), type=int, help="Max examples per file")
    parser.add_argument('--num_files', default=None, type=int, help="Only read n files from file")
    parser.add_argument('--no_parallel', action='store_true', default=False, help='Do not run in parallel')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
