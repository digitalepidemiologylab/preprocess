import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder
import logging


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
    df = get_all_data(dtype=dtype, include_flags=True, include_predictions=False)
    
    # fine tune data should
    # - not contain any training data
    # - not be a text duplicate
    # - have min 3 tokens
    # - should not be a retweet
    logger.info('Filtering...')
    df = df[(~df.A) & (~df.is_duplicate) & (df.token_count >= 3) & (~df.is_retweet) & (df.contains_keywords)]

    # write output file
    f_path = os.path.join(find_folder('1_parsed'), 'parsed_{}_finetune.csv'.format(dtype))
    logger.info('Writing {:,} lines to file {}...'.format(len(df), f_path))
    df[['text']].to_csv(f_path, index=False)


if __name__ == "__main__":
    main()
