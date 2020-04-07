import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder
import logging
from tqdm import tqdm

transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])

def main(dtype='anonymized'):
    """
    This script creates a new file `data/1_parsed/parsed_{dtype}_finetune.csv` containing data to fine-tune a text classification model.
    It excludes training data and duplicates.
    """
    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    logger = logging.getLogger(__name__)

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

    # standardize text
    logger.info('Standardize text...')
    df.loc[:, 'text'] = df.text.apply(standardize_text)

    # remove duplicates (after removing user/url tag)
    f_path = os.path.join(find_folder('1_parsed'), 'parsed_{}_finetune.txt'.format(dtype))
    df['text_cleared'] = df.text.str.replace(r'@<user>|<url>', '')
    df['text_cleared'] = df.text_cleared.str.strip()
    df['text_cleared'] = df.text_cleared.str.lower()
    df = df.drop_duplicates(subset=['text_cleared'])
    # write output file
    logger.info('Writing {:,} lines to file {}...'.format(len(df), f_path))
    with open(f_path, 'w') as f:
        for i, text in tqdm(df.text.iteritems(), total=len(df)):
            f.write(text + '\n')


def standardize_text(text):
    """Replace some non-standard characters such as ” or ’ with standard characters. """
    text = text.translate(transl_table)
    # Sometimes user handles seem to contain an additional @ which will lead to @@<user> during anonymization
    text = text.replace('@@<user>', '@<user>')
    return text

if __name__ == "__main__":
    main()
