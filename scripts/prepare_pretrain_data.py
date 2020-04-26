import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder
from utils.misc import ArgParseDefault
import logging
from tqdm import tqdm, trange
import re
import joblib
import multiprocessing
import datetime


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
    usecols = ['id', 'text', 'lang', 'is_duplicate', 'token_count', 'is_retweet', 'contains_keywords']
    df = get_all_data(dtype=args.dtype, include_flags=True, include_predictions=False, include_cleaned_labels=False, usecols=usecols, nrows=args.nrows)
    logger.info(f'...loaded a total of {len(df):,} tweets')

    filters = {'contains_keywords': 'contains_keywords'}
    filters_not = {'retweets': 'is_retweet', 'training_data': 'L', 'is_duplicate': 'is_duplicate'}

    # Filtering by col
    for _filter in args.filters:
        logger.info(f'Applying filter {_filter}')
        num_before = len(df)
        if _filter in filters:
            df = df[df[filters[_filter]]]
        elif _filter in filters_not:
            df = df[~df[filters_not[_filter]]]
        else:
            raise ValueError(f'Unknown filter {_filter}')
        num_after = len(df)
        logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    logger.info(f'Filter lang {args.lang}...')
    num_before = len(df)
    df = df[df.lang == args.lang]
    num_after = len(df)
    logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    logger.info(f'Filter has >={args.min_tokens} tokens...')
    df = df[df.token_count >= args.min_tokens]
    num_after = len(df)
    logger.info(f'... {num_after:,} remaining (removed {num_before-num_after:,})')

    if args.dtype == 'anonymized':
        logger.info('Replacing user handles...')
        # make sure there are no user handles left
        df.loc[:, 'text'] = df.text.apply(replace_user_handles)

    # standardize text
    logger.info('Standardize text...')
    df.loc[:, 'text'] = df.text.apply(standardize_text)

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
            os.path.join(output_folder, f'pretrain_{args.dtype}_{_type}_{j:03}.txt')
            ) for j, i in enumerate(trange(0, len(_df), args.max_examples_per_file))))
        logger.info(f'Successfully wrote {len(res):,} file(s) to folder {output_folder}')

def write_output_file(df, f_path):
    with open(f_path, 'w') as f:
        num_lines = len(df)
        for i, text in tqdm(df.text.iteritems(), total=num_lines):
            f.write(text + '\n')
    return 1

def generate_text_cleared(text):
    # Remove RT @<user>, @<user> and <url>
    text = re.sub(r'RT @<user>:|@<user>|<url>', '', text)
    # Remove induced whitespaces
    text = ' '.join(text.split())
    # lowercase everything
    text = text.lower()
    return text

def standardize_text(text):
    """Replace some non-standard characters such as ” or ’ with standard characters. """
    text = text.translate(transl_table)
    # Sometimes user handles seem to contain an additional @ which will lead to @@<user> during anonymization
    text = text.replace('@@<user>', '@<user>')
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def replace_user_handles(text):
    text = re.sub(user_handle_regex, '@<user>', text)
    return text

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--filters', choices=['retweets', 'training_data', 'is_duplicate', 'contains_keywords'], default=['retweets', 'training_data', 'is_duplicate'], help="Apply filters")
    parser.add_argument('--dtype', default='anonymized', choices=['original', 'anonymized', 'encrypted'], help="Data type")
    parser.add_argument('--lang', default='en', help="Filter language")
    parser.add_argument('--min_tokens', default=3, help="Min num tokens")
    parser.add_argument('--max_examples_per_file', default=int(1e6), type=int, help="Max examples per file")
    parser.add_argument('--nrows', default=None, type=int, help="Only read n rows from file")
    parser.add_argument('--no_parallel', action='store_true', default=False, help='Do not run in parallel')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
