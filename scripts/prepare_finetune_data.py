import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder
import logging
from tqdm import tqdm
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
user_handle_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

def main(dtype='anonymized'):
    """
    This script creates a new file `data/1_parsed/parsed_{dtype}_finetune{train/dev}.csv` containing data to fine-tune a text classification model.
    It excludes training data and duplicates.
    """
    # load data
    logger.info('Reading data...')
    usecols = ['id', 'text', 'is_duplicate', 'token_count', 'is_retweet', 'contains_keywords']
    df = get_all_data(dtype=dtype, include_flags=True, include_predictions=False, usecols=usecols)

    # fine tune data should
    # - not contain any training data
    # - not be a text duplicate
    # - have min 3 tokens
    # - should not be a retweet
    logger.info('Filtering...')
    df = df[(~df.A) & (~df.is_duplicate) & (df.token_count >= 3) & (~df.is_retweet) & (df.contains_keywords)]

    if dtype == 'anonymized':
        logger.info('Replacing user handles...')
        # make sure there are no user handles left
        df.loc[:, 'text'] = df.text.apply(replace_user_handles)

    # standardize text
    logger.info('Standardize text...')
    df.loc[:, 'text'] = df.text.apply(standardize_text)

    # generate text column to filter for duplicates
    logger.info('Remove duplicates...')
    df.loc[:, 'text_cleared'] = df.text.apply(generate_text_cleared)
    df = df.drop_duplicates(subset=['text_cleared'])

    # shuffle
    logger.info('Shuffle...')
    df = df.sample(frac=1)

    # write to train/dev file
    num_lines = len(df)
    logger.info(f'Collected total of {num_lines:,} examples')
    num_train = min(int(0.8*num_lines), int(2e5))
    for (_s, _e), _type in zip([(None, num_train), (num_train, None)], ['train', 'dev']):
        f_path = os.path.join(find_folder('1_parsed'), f'parsed_{dtype}_finetune_{_type}.txt')
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

if __name__ == "__main__":
    main()
