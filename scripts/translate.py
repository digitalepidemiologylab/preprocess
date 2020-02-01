import argparse
import logging
import sys, os
import requests
import pandas as pd
sys.path.append('..')
from utils.helpers import get_parsed_data, find_folder
from utils.misc import get_df_hash
import glob
from tqdm import tqdm



USAGE_DESC = """
python main.py <command> [<args>]

Available commands:
  prepare             Prepares data for translation
  translate           Translates prepared data
"""

class ArgParseDefault(argparse.ArgumentParser):
    """Simple wrapper which shows defaults in help"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class ArgParse(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
        parser = ArgParseDefault(
                description='',
                usage=USAGE_DESC)
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        getattr(self, args.command)()

    def prepare(self):
        parser = ArgParseDefault(description='Prepare text to be translated')
        logger = logging.getLogger(__name__)
        parser.add_argument('--geo-only', dest='geo_only', action='store_true', help='Only use geo-tagged data')
        parser.add_argument('-d', '--dtype', dest='dtype', choices=['original', 'anonymized', 'encrypted'], default='anonymized', help='Data source type')
        parser.add_argument('-l', '--limit', dest='limit', type=int, default=-1, help='If set, only extract random subsample')
        args = parser.parse_args(sys.argv[2:])
        # load data
        logger.info('Loading data...')
        df = get_parsed_data(dtype=args.dtype, usecols=['id', 'text', 'is_duplicate', 'has_place', 'has_coordinates', 'is_retweet'])
        # filter
        if args.geo_only:
            df = df[(df.has_place | df.has_coordinates) & (~df.is_duplicate) & (~df.is_retweet)]
        else:
            df = df[(~df.is_duplicate) & (~df.is_retweet)]
        if args.limit > 0:
            df = df.sample(args.limit)
        # write data
        folder = os.path.join(find_folder('other'), 'translations')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        f_path = os.path.join(folder, 'prepare_{}.csv'.format(get_df_hash(df)[:5]))
        logger.info('Writing {:,} records to file {}...'.format(len(df), f_path))
        df[['id', 'text']].to_csv(f_path, index=False)

    def translate(self):
        parser = ArgParseDefault(description='Translate prepared text')
        logger = logging.getLogger(__name__)
        parser.add_argument('-s', '--source', dest='source', choices=["EN", "DE", "FR", "ES", "PT", "IT", "NL", "PL", "RU"], required=True, help='Source language')
        parser.add_argument('-t', '--target', dest='target', choices=["EN", "DE", "FR", "ES", "PT", "IT", "NL", "PL", "RU"], required=True, help='Target language')
        parser.add_argument('--auth-key', dest='auth_key', type=str, required=True, help='DeepL auth key')
        args = parser.parse_args(sys.argv[2:])
        logger.info('Translating from source language {} to target language {}....'.format(args.source, args.target))
        # load data
        folder = os.path.join(find_folder('other'), 'translations')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        f_paths = glob.glob(os.path.join(folder, 'prepare_*.csv')) 
        if len(f_paths) == 0:
            raise FileNotFoundError('No prepare_*.csv file(s) found in folder {}'.format(folder))
        elif len(f_paths) > 1:
            raise ValueError('Found {} prepare_*.csv files in folder {}.'.format(len(f_paths), folder))

        logger.info('Loading prepared data...')
        df = pd.read_csv(f_paths[0], dtype={'id': str})
        df_len = df.text.apply(len)
        costs = df_len.sum()/500 * 0.01
        logger.info('About to translate {:,} characters with an estimated cost of EUR {:.2f}.'.format(df_len.sum(), costs))
        yes_no = input('Continue to translate? (yes/no)\n')
        if not (yes_no == 'y' or yes_no == 'yes'):
            logger.info('Aborting...')
            return

        # params
        base_url = 'https://api.deepl.com/v2/translate'
        other_params = {'auth_key': args.auth_key, 'target_lang': args.target, 'source_lang': args.source}
        chunk_size = 20
        def chunks(total_len, n):
            for i in range(0, total_len, n):
                yield i, i + n - 1
        df['translation'] = ''
        for start, stop in tqdm(chunks(len(df), chunk_size), total=len(range(0, len(df), chunk_size))):
            texts = df.loc[start:stop, 'text'].tolist()
            res = requests.get(base_url, params={'text': texts, **other_params})
            if not res.ok:
                raise Exception('Unable to retrieve data from DeepL. Error status code {}... Aborting'.format(res.status_code))
            res = res.json()
            df_tr = pd.DataFrame(res['translations'])
            df.loc[start:stop, 'translation'] = df_tr['text'].values
        f_path = os.path.join(folder, 'translation_{}.csv'.format(get_df_hash(df)[:5]))
        print('Writing {:,} records to file {}...'.format(len(df), f_path))
        df.to_csv(f_path, index=False)


if __name__ == '__main__':
    ArgParse()
