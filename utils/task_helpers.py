import logging
from collections import Counter
import os
import json
from utils.helpers import find_folder, get_cleaned_labelled_data, get_project_info, find_project_root, get_parsed_data
from utils.s3_helper import S3Helper
from utils.stats import Stats
from utils.misc import JSONEncoder
from utils.process_tweet import ProcessTweet
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import joblib
import multiprocessing
from tqdm import tqdm
import csv

logger = logging.getLogger(__name__)

def init(project, template):
    project_fname = os.path.join(find_project_root(), 'project_info.json')
    logger = logging.getLogger(__name__)
    # write empty template file to fill in manually
    if template:
        template = {"name": project, "keywords": []}
        with open(project_fname, 'w') as f:
            json.dump(template, f, cls=JSONEncoder, indent=4)
        logger.info('Successfully wrote empty template file "{}". Please fill in values manually.'.format(project_fname))
        return
    # sync project info
    s3_helper = S3Helper()
    s3_helper.sync_project_info(project)


def train_dev_test_split(question='sentiment', dev_size=0.1, test_size=0.2, seed=42, name='', balanced_labels=False, all_questions=False, label_tags=[], labelled_as=None, has_label=''):
    """Splits cleaned labelled data into training, dev and test set"""
    def _filter_for_label_balance(df):
        """Performs undersampling for overrepresanted label classes"""
        counts = Counter(df['label'])
        min_count = min(counts.values())
        _df = pd.DataFrame()
        for l in counts.keys():
            _df = pd.concat([_df, df[df['label'] == l].sample(min_count)])
        return _df
    questions = [question]
    np.random.seed(seed)
    if name == '':
        f_path = os.path.join(find_folder('4_labels_cleaned'), 'cleaned_labels*.csv')
        annotation_files = glob.glob(f_path)
        if len(annotation_files) == 0:
            raise FileNotFoundError(f'No cleaned label files could be found with the pattern {f_path}')
        elif len(annotation_files) > 1:
            raise ValueError(f'Found {len(annotation_files)} different files for cleaned labels. Provide "name" argument to specify which.')
        name = os.path.basename(annotation_files[0]).split('.csv')[0]
    if all_questions:
        df = get_cleaned_labelled_data(name=name)
        questions = df['question_tag'].unique()
    for question in questions:
        df = get_cleaned_labelled_data(question=question, name=name, has_label=has_label)
        if len(df) == 0:
            logger.warning('No labelled data could be found for question {} under these parameters.'.format(question))
            continue
        if balanced_labels:
            df = _filter_for_label_balance(df)
        flags = '{}{}'.format('_' + name if name != '' else '', '_balanced' if balanced_labels else '')
        if len(label_tags) > 0:
            df = df[df['label'].isin(label_tags)]
            flags += '_labels_{}'.format('_'.join(label_tags))
        if len(has_label) > 0:
            has_label_flag = 'has_label_{}'.format(has_label.replace('|', '_or_').replace(',', '_and_'))
            flags += '_' + has_label_flag
            folder_path = os.path.join(find_folder('4_labels_cleaned'), 'other', has_label_flag, question)
        else:
            folder_path = os.path.join(find_folder('4_labels_cleaned'), 'splits', question)
        train, dev, test = np.split(df.sample(frac=1, random_state=seed), [int((1-dev_size-test_size)*len(df)), int((1-test_size)*len(df))])
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        for dtype, data in [['train', train], ['dev', dev], ['test', test]]:
            f_name = f'{dtype}_{question}_split_{len(train)}_{len(dev)}_{len(test)}_seed_{seed}{flags}.csv'
            f_path = os.path.join(folder_path, f_name)
            data.to_csv(f_path, index=None, encoding='utf8')
            logger.info(f'Successfully wrote data of {len(data):,} examples to file {f_path}.')

def sync(data_type='all', last_n_days=None):
    project_info = get_project_info()
    project_name = project_info['name']
    s3_helper = S3Helper()
    s3_helper.sync(project_name, data_type=data_type, last_n_days=last_n_days)

def stats(stats_type, **args):
    """Output general info about project"""
    stats = Stats()
    if stats_type == 'overview':
        stats.overview()
    elif stats_type == 'all':
        stats.all()
    elif stats_type == 'sample':
        stats.sample()
    elif stats_type == 'annotation':
        stats.annotation(**args)
    elif stats_type == 'annotator_outliers':
        stats.annotator_outliers(**args)
    elif stats_type == 'annotation_cleaned':
        stats.annotation_cleaned(**args)
    else:
        raise ValueError('Command {} is not available.'.format(stats_type))
    print(stats.text)


def prepare_predict(args):
    # paths
    date_key = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    f_out_folder = os.path.join(find_project_root(), 'data', 'other', 'prepare_prediction', date_key)
    f_path_txt = os.path.join(f_out_folder, 'text.csv')
    f_path_meta = os.path.join(f_out_folder, 'id_created_at.csv')
    if os.path.isdir(f_out_folder):
        raise Exception(f'Folder {f_out_folder} already exists')
    os.makedirs(f_out_folder)
    # read data
    logger.info('Reading raw data...')
    df = get_parsed_data(num_files=None, s_date=args.start_date, e_date=args.end_date, usecols=['id', 'created_at', 'text'])
    if args.start_date is not None:
        df = df[df.created_at >= args.start_date]
    if args.end_date is not None:
        df = df[df.created_at <= args.end_date]
    logger.info('Sorting...')
    df = df.sort_values('created_at')
    if args.anonymize:
        def anonymize(df_chunk):
            return df_chunk.apply(ProcessTweet.anonymize_text, url_filler=args.url_filler, user_filler=args.user_filler)

        logger.info('Anonymize...')
        anonymize_delayed = joblib.delayed(anonymize)
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
        parallel = joblib.Parallel(n_jobs=num_cores)
        res = parallel(anonymize_delayed(df_chunk) for df_chunk in tqdm(np.array_split(df['text'], num_cores), unit='chunk'))
        df['text'] = pd.concat(res)
    # write data
    logger.info(f'Writing text column to {f_path_txt}...')
    df[['text']].to_csv(f_path_txt, index=False, header=False)#, quoting=csv.QUOTE_NONE, escapechar="\\")
    logger.info(f'Writing id/created_at column to {f_path_meta}...')
    df[['id', 'created_at']].to_csv(f_path_meta, index=False)
