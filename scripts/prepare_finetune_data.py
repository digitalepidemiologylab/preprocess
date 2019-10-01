import pandas as pd
import os
import sys
sys.path.append('..')
from utils.helpers import get_all_data, find_folder



def main():
    """
    This script creates a new file `data/2_cleaned/2_cleaned_finetune.csv` containing data to fine-tune a text classification model.
    It excludes training data and duplicates.
    """
    # load data
    print('Reading data...')
    df = get_all_data(dtype='anonymized', include_flags=True, include_predictions=False)

    # fine tune data should
    # - not contain any training data
    # - not be a text duplicate
    # - have min 3 tokens
    # - should not be a retweet
    print('Filtering...')
    df = df[(~df.A) & (~df.is_duplicate) & (df.token_count >= 3) & (~df.is_retweet)]

    # write output file
    f_path = os.path.join(find_folder('2_cleaned'), 'cleaned_labels_finetune.csv')
    print('Writing file {}...'.format(f_path))
    df[['text']].to_csv(f_path, index=False)


if __name__ == "__main__":
    main()
