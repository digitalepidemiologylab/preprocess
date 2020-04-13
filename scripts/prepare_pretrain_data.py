import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder
from utils.misc import ArgParseDefault
import logging
from tqdm import tqdm
import re
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
user_handle_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

def main(args):
    """
    This script creates a new file `data/1_parsed/parsed_{dtype}_finetune{train/dev}.csv` containing data to fine-tune a text classification model.
    It excludes training data and duplicates.
    """
    # load data
    logger.info('Reading data...')
    usecols = ['id', 'text', 'lang', 'is_duplicate', 'token_count', 'is_retweet', 'contains_keywords']
    df = get_all_data(dtype=args.dtype, include_flags=True, include_predictions=False, include_cleaned_labels=False, usecols=usecols)
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

    # write to train/dev file
    num_lines = len(df)
    logger.info(f'Collected total of {num_lines:,} examples')
    num_train = max(int(0.8*num_lines), num_lines - int(2e5))
    for (_s, _e), _type in zip([(None, num_train), (num_train, None)], ['train', 'dev']):
        f_path = os.path.join(find_folder('1_parsed'), f'parsed_{args.dtype}_pretrain_{_type}.txt')
        num_lines = len(df[_s:_e])
        logger.info(f'Writing {num_lines:,} examples to file {f_path}...')
        with open(f_path, 'w') as f:
            for i, text in tqdm(df[_s:_e].text.iteritems(), total=num_lines):
                f.write(text + '\n')

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
    # text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', '@<user>', text)
    text = re.sub(user_handle_regex, '@<user>', text)
    return text


def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--filters', choices=['retweets', 'training_data', 'is_duplicate', 'contains_keywords'], default=['retweets', 'training_data', 'is_duplicate'], help="Apply filters")
    parser.add_argument('--dtype', default='anonymized', choices=['original', 'anonymized', 'encrypted'], help="Data type")
    parser.add_argument('--lang', default='en', help="Filter language")
    parser.add_argument('--min_tokens', default=3, help="Min num tokens")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
