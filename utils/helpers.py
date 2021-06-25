import os
import re
import io
import glob

import pandas as pd
import numpy as np
import json
import warnings
import csv
import logging
import multiprocessing
import joblib
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)


def get_parsed_data(contains_keywords=False, usecols=None, s_date=None, e_date=None, read_in_parallel=True, num_files=None):
    """Read parsed data
    :param contains_keywords: Only get data where contains_keywords is True (default: False)
    :param usecols: Only extract certain columns (default: all columns)
    :param s_date: start date to read from (str of format YYYY-MM-dd or datetime obj)
    :param e_date: end date to read from (str of format YYYY-MM-dd or datetime obj)
    """
    def parse_date_from_f_name(f_name):
        f_name = os.path.basename(f_name)
        date = f_name.split('.')[0][len('parsed_'):]
        if len(date.split('-')) == 3:
            # YYYY-MM-dd format
            return datetime.strptime(date, '%Y-%m-%d')
        else:
            # assume YYYY-MM-dd-HH format
            return datetime.strptime(date, '%Y-%m-%d-%H')
    def get_f_names(s_date=s_date, e_date=e_date):
        data_folder = get_data_folder()
        f_path = os.path.join(data_folder, '1_parsed', 'tweets', '*.parquet')
        f_names = glob.glob(f_path)
        f_names = sorted(f_names)
        if s_date is not None or e_date is not None:
            f_names_dates = {f_name: parse_date_from_f_name(f_name) for f_name in f_names}
            if s_date is not None:
                if not isinstance(s_date, datetime):
                    s_date = datetime.strptime(s_date, '%Y-%m-%d')
                f_names = [f_name for f_name in f_names if f_names_dates[f_name] > s_date]
            if e_date is not None:
                if not isinstance(e_date, datetime):
                    e_date = datetime.strptime(e_date, '%Y-%m-%d')
                f_names = [f_name for f_name in f_names if f_names_dates[f_name] < e_date]
        return f_names

    def read_data(f_name):
        """Reads single parquet file"""
        df = pd.read_parquet(f_name, columns=usecols)
        if contains_keywords:
            df = df[df.contains_keywords].drop(columns=['contains_keywords'])
        return df
    # sanity check on params
    if contains_keywords and usecols is not None and 'contains_keywords' not in usecols:
        usecols += ['contains_keywords']
    # load files
    f_names = get_f_names()
    if isinstance(num_files, int):
        f_names = f_names[:num_files]
    # set up parallel
    if read_in_parallel:
        n_jobs = max(multiprocessing.cpu_count() - 1, 1)
    else:
        n_jobs = 1
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer='threads')
    read_data_delayed = joblib.delayed(read_data)
    # load data
    logger.info('Reading data...')
    df = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    logger.info('Concatenating data...')
    df = pd.concat(df)
    # convert to category
    for col in ['country_code', 'region', 'subregion', 'geo_type', 'lang']:
        if col in df:
            df[col] = df[col].astype('category')
    return df

def get_labelled_data(pattern='*', mode='*', usecols=None):
    """Read labelled data
    :param pattern: Only select a specific file by using a pattern (default: *)
    :param mode: Can either be 'local', 'mturk' or 'public', 'other' (default: *)
    """
    accepted_modes = ['*', 'all', 'local', 'mturk', 'public', 'other']
    if mode not in accepted_modes:
        raise ValueError('Mode argument has to be one of: {}'.format(', '.join(accepted_modes)))
    if mode == 'all':
        mode = '*'
    f_names = glob.glob(os.path.join(find_folder('3_labelled'), mode, '*{}*.csv'.format(pattern)))
    if len(f_names) == 0:
        raise FileNotFoundError('No annotation files could be found.')
    df = pd.DataFrame()
    # Make sure to keep id columns as pandas integer type Int64 which also allows for NaN values
    dtypes = {'tweet_id': str, 'question_id': 'Int64', 'answer_id': 'Int64', 'project_id': 'Int64', 'user_id': 'Int64'}
    for f_path in f_names:
        df = pd.concat([df, pd.read_csv(f_path, dtype=dtypes, usecols=usecols)], axis=0, ignore_index=True, sort=True)
    # transform
    if 'full_log' in df:
        log_not_null = df.full_log.notnull()
        df_new = df[log_not_null].full_log.apply(json.loads)
        df.update(df_new)
    if 'user_id' in df or 'worker_id' in df:
        df['annotator_id'] = np.nan
        for annotator in ['user_id', 'worker_id']:
            if annotator in df:
                df['annotator_id'] = df['annotator_id'].fillna(df[annotator])
    if 'answer_tag' in df and 'question_tag' in df:
        null_answer_tags = sum(df.answer_tag.isnull())
        null_question_tags = sum(df.question_tag.isnull())
        if null_answer_tags > 0:
            logger.warning('The label dataset contains {:,} annotations without an answer tag.'.format(null_answer_tags))
        if null_question_tags > 0:
            logger.warning('The label dataset contains {:,} annotations without a question tag.'.format(null_question_tags))
    return df


def get_cleaned_labelled_data(question=None, name='', cols=None, return_label_ids=False, has_label=''):
    """Read cleaned labelled data
    :param question: if int get quesiton_id, if string must be a valid question tag
    :param name: Name of cleaned_labels file (default: '')
    :param cols: Columns to return, can contain (id|text|label|question)
    :param return_label_ids: Return label ID instead of label tag (default: False)
    :param has_label: Filter for tweets which have been annotated with this label. For multiple labels use | as OR seperator and , for AND (default: '').
    """
    if name == '':
        f_path = os.path.join(find_folder('4_labels_cleaned'), 'cleaned_labels*.csv')
    else:
        f_path = os.path.join(find_folder('4_labels_cleaned'), '{}.csv'.format(name))
    annotation_files = glob.glob(f_path)
    if len(annotation_files) == 0:
        raise FileNotFoundError('No cleaned label files could be found with the pattern {}'.format(f_path))
    elif len(annotation_files) > 1:
        raise ValueError('Found {} different files for cleaned labels. Provide "name" argument to specify which.'.format(len(annotation_files)))
    dtypes = {'id': str, 'question_id': 'Int64', 'answer_id': 'Int64'}
    df = pd.DataFrame()
    for annotation_file in annotation_files:
        df_annot = pd.read_csv(annotation_file, encoding='utf8', dtype=dtypes)
        df = pd.concat([df, df_annot], axis=0)
    possible_question_tags = df.question_tag.unique()
    if has_label is not None and has_label != '':
        for _has_label in has_label.split(','):
            filter_has_label = []
            for tag in _has_label.split('|'):
                filter_has_label.append('^{}$'.format(tag))
            filter_has_label = '|'.join(filter_has_label)
            relevant_ids = df[df['label_tag'].str.contains(filter_has_label)]['id']
            df = df.loc[df['id'].isin(relevant_ids)]
    if len(df) == 0:
        return df
    if question == '' or question is None:
        if cols is None:
            cols = ['id', 'text', 'question_tag', 'label']
    else:
        if isinstance(question, str):
            # assume question is a tag and not an ID
            question_ids = set(df[df['question_tag'] == question]['question_id'])
            if len(question_ids) == 0:
                if question in possible_question_tags:
                    return pd.DataFrame()
                else:
                    possible_question_tags_str = ', '.join(['"{}"'.format(q) for q in possible_question_tags])
                    raise ValueError('Question `{}` could not be found. The following question tags are supported: {}'.format(question, possible_question_tags_str))
            elif len(question_ids) > 1:
                logger.warning('Question `{}` maps to multiple question IDs and is therefore not unique.'.format(question))
            df.rename(columns={'question_tag': 'question'}, inplace=True)
        else:
            df.rename(columns={'question_id': 'question'}, inplace=True)
        df = df[df['question_id'].isin(question_ids)]
        if cols is None:
            cols = ['id', 'text', 'label']
    if return_label_ids:
        df.rename(columns={'label_id': 'label'}, inplace=True)
    else:
        df.rename(columns={'label_tag': 'label'}, inplace=True)
    return df[cols]

def get_sampled_data(dtype='*', usecols=None):
    """Read sampled data
    :param dtype: possible values: "original", "anonymized", "encrypted", default: read all sampled data
    """
    f_names = glob.glob(os.path.join(find_folder('2_sampled'), 'sampled_{}_*.csv'.format(dtype)))
    if len(f_names) == 0:
        raise FileNotFoundError('No data files could be found')
    elif len(f_names) != 1:
        logger.info('Reading from {} sample files and merging.'.format(len(f_names)))
    df = pd.DataFrame()
    dtypes = {'tweet_id': str, 'tweet_text': str}
    for f_path in f_names:
        df = df.append(pd.read_csv(f_path, header=None, names=['tweet_id', 'tweet_text'], dtype=dtypes, index_col=None, usecols=usecols))
    return df

def get_batched_sample_data():
    df = pd.DataFrame()
    f_names = glob.glob(os.path.join(find_folder('2_sampled'), 'batch_*', 'batch_*.csv'))
    for f_n in f_names:
        df = df.append(pd.read_csv(f_n, header=None, names=['tweet_id', 'tweet_text'], dtype={'tweet_id': str, 'tweet_text': str}, index_col=None))
    return df

def get_uploaded_batched_data(availability=None, mode='*'):
    possible_modes = ['*', 'local', 'mturk', 'public', 'other']
    if mode not in possible_modes:
        raise ValueError('Parameter mode is {} but has to be one of {}'.format(mode, ', '.join(possible_modes)))
    availabilities = [None, 'available', 'unavailable']
    if availability not in availabilities:
        raise ValueError('Parameter availability is {} but has to be one of {}'.format(availability, ', '.join(availabilities)))
    df = pd.DataFrame()
    f_names = glob.glob(os.path.join(find_folder('3_labelled'), mode, 'batches', '*.csv'))
    for f_n in f_names:
        df = df.append(pd.read_csv(f_n, index_col=None, dtype={'tweet_id': str, 'tweet_text': str}))
    if len(df) == 0:
        return set()
    if availability is not None:
        try:
            df = df.groupby('availability').get_group(availability)
        except KeyError:
            return set()
    return df

def get_checked_tweets(mode='*', availability='*', df=None):
    possible_modes = ['*', 'local', 'mturk', 'public', 'other']
    if mode not in possible_modes:
        raise ValueError('Parameter mode is {} but has to be one of {}'.format(mode, ', '.join(possible_modes)))
    df_checked = pd.DataFrame()
    f_names = glob.glob(os.path.join(find_folder('3_labelled'), mode, 'batches', '*.csv'))
    for f_n in f_names:
        df_checked = df_checked.append(pd.read_csv(f_n, index_col=None, dtype={'tweet_id': str, 'tweet_text': str}))
    df_checked = df_checked[['tweet_id', 'availability']]
    if availability != '*':
        df_checked = df_checked[df_checked.availability==availability]
    if not df is None:
        df_checked = pd.merge(df, df_checked, how='inner', on='tweet_id')
        df_checked.set_index('created_at', inplace=True)
    return df_checked

def get_predicted_data(include_raw_data=True, dtype='anonymized', flag=None, drop_retweets=False, usecols=None):
    """
    Return prediction data. Prediction data should be in `data/5_predicted`. All file names should have the following pattern: `predicted_{column_name}_{YYY}-{MM}-{dd}_{5-char-hash}.csv`.
    The column name can be an arbitrary tag (e.g. question tag) which should be unique.
    :param include_raw_data: Merge predictions with 1_parsed data (by default True)
    :param dtype: Raw data dtype (default: anonymized), relevant if include_raw_data is set to True.
    :param flag: Filter raw data by certain flags, relevant if include_raw_data is set to True.
    :param drop_retweets: Drop retweets (default: False), relevant if include_raw_data is set to True.
    :param usecols: Only select certains columns from 1_parsed data (by default all), relevant if include_raw_data is set to True.
    """
    f_pattern = os.path.join(find_folder('5_predicted'), 'predicted_*.csv')
    f_names = glob.glob(f_pattern)
    if len(f_names) == 0:
        raise FileNotFoundError('No prediction files present with pattern {}'.format(f_pattern))
    df_pred = pd.DataFrame()
    df_column_names = {}
    # map file names to unique column names in df
    f_name_pattern = re.compile(r'predicted_(.+)_\d{4}-\d{2}-\d{2}_.{5}.csv')
    for f_name in f_names:
        basename = os.path.basename(f_name)
        matches = f_name_pattern.match(basename)
        if bool(matches):
            df_column_names[f_name] = matches.groups()[0]
        else:
            # simple use prefix of filename
            df_column_names[f_name] = basename[len('predicted_'):-len('.csv')]
    for f_name in f_names:
        df = pd.read_csv(f_name)
        column_rename = {}
        for c in ['label', 'probability', 'labels', 'probabilities']:
            column_rename[c] = '{}_{}'.format(c, df_column_names[f_name])
        df.rename(columns=column_rename, inplace=True)
        df_pred = pd.concat([df_pred, df], axis=1)
    if include_raw_data:
        df = get_parsed_data(usecols=usecols)
        assert len(df) == len(df_pred), 'Length of prediction and raw data are not equal'
        df_pred.index = df.index
        df = pd.concat([df, df_pred], axis=1, sort=True)
        if flag is not None:
            df = df[df[flag]]
        if drop_retweets:
            df = df[~df['is_retweet']]
        return df
    else:
        return df_pred

def get_all_data(include_all_data=False, include_cleaned_labels=True, usecols=None, extra_cols=None, dtype='anonymized', s_date='', e_date='', mode='*', include_predictions=True, include_flags=False, geo_enrichment_type=None):
    """
    Returns all data including predictions and optionally certain flags
    :param include_all_data: If set to False return the minimal possible number of columns (id, predictions, filters), default: True
    :param dtype: Raw data dtype (default: anonymized)
    :param s_date: Start date filter (YYYY-MM-dd)
    :param e_date: End date filter (YYYY-MM-dd)
    :param include_flags: Include certain flags (S: used in sampling, L: labelled, A: cleaned labelled)
    :param mode: Annotation mode (*/mturk/local/public/other). Only relevant when include_flags set to true
    :param include_predictions: Include all model predictions
    :param geo_enrichment_type: Can be None (do not include any enrichment data) or 'all'. If set, will include inferred geo data
    """
    # load data
    if usecols is None:
        # default columns
        usecols = ['id', 'token_count', 'is_retweet', 'contains_keywords', 'created_at' , 'lang']
        if extra_cols is not None:
            for ec in extra_cols:
                usecols.append(ec)
    df = get_parsed_data(usecols=usecols)
    if include_predictions:
        df_pred = get_predicted_data(include_raw_data=False, dtype=dtype)
        df_pred.index = df.index
        df = pd.concat([df, df_pred], axis=1)
    # compute filters for raw data
    if s_date != '':
        df = df[df.index >= s_date]
    if e_date != '':
        df = df[df.index <= e_date]
    if include_flags:
        # compute filters for sampling data
        df_sampled = get_uploaded_batched_data(mode=mode)
        df['S'] = df.id.isin(df_sampled)
        # compute filters for annotation data
        df_labelled = get_labelled_data(usecols=['tweet_id'], mode=mode)
        df['L'] = df.id.isin(df_labelled.tweet_id)
        if include_cleaned_labels:
            df_cleaned_labels = get_cleaned_labelled_data(name='*', cols=['id'])
            df['A'] = df.id.isin(df_cleaned_labels.id)
    if geo_enrichment_type is not None:
        cache_path = get_cache_path(f'geonames_enriched_{dtype}.pkl')
        df_geo = pd.read_pickle(cache_path)
        df = df.reset_index()
        df = pd.concat([df, df_geo], axis=1)
        df = df.set_index('created_at')
    return df

def get_dtypes(usecols=None):
    """Gets dtypes for columns"""
    dtypes = {
            "id": str, 
            "text": str, 
            "in_reply_to_status_id": str,
            "in_reply_to_user_id": str,
            "quoted_status_id": str,
            "quoted_user_id": str,
            "retweeted_status_id": str,
            "retweeted_user_id": str,
            "created_at": str,
            "entities.user_mentions": str,
            "user.id": str,
            "user.screen_name": str,
            "user.name": str,
            "user.description": str,
            "user.timezone": str,
            "user.location": str,
            "user.num_followers": int,
            "user.num_following": int,
            "user.created_at": str, 
            "user.statuses_count": int,
            "user.is_verified": bool,
            "lang": str,
            "token_count": int,
            "is_retweet": bool,
            "has_quote": bool,
            "is_reply": bool,
            "contains_keywords": bool,
            "longitude": float,
            "latitude": float,
            "country_code": str,
            "region": str,
            "subregion": str,
            "location_type": str,
            "geoname_id": str,
            "geo_type": int
            }
    if usecols is not None:
        dtypes = {i:v for i,v in dtypes.items() if i in usecols}
    return dtypes

def get_f_name(dtype, processing_step='merged', flag=None, contains_keywords=False, fmt='csv'):
    if flag is None:
        flag = ''
    else:
        flag = '_' + flag
    if contains_keywords:
        contains_keywords_flag = '_contains_keywords'
    else:
        contains_keywords_flag = ''
    f_name = f'{processing_step}_{dtype}{flag}{contains_keywords_flag}.{fmt}'
    return f_name

def find_file(f_name, subfolder='1_merged', cached=False):
    f_path = os.path.join(find_folder(subfolder), f_name)
    if os.path.isfile(f_path):
        return f_path
    if not cached:
        raise FileNotFoundError('File {0} could not be found.'.format(f_name))
    else:
        return ''

def find_folder(folder_path, subfolder='data', num_par_dirs=4):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if subfolder == '.':
        # ignore subfolder if current folder
        subfolder = ''
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        f_path = os.path.join(current_folder, *par_dirs, subfolder, folder_path)
        if os.path.isdir(f_path):
            return os.path.abspath(f_path)
    raise FileNotFoundError('Folder {0} could not be found.'.format(folder_path))

def get_project_info():
    project_fname = os.path.join(find_project_root(), 'project_info.json')
    if not os.path.isfile(project_fname):
        raise FileNotFoundError('Project info file {} could not be found in root folder. Run the init command first.'.format(project_fname))
    with open(project_fname, 'r') as f:
        project_info = json.load(f)
    for required_key, f_type in {'keywords': list, 'name': str}.items():
        if required_key not in project_info:
            raise ValueError('Project info file does not contain key {}'.format(required_key))
        if not isinstance(project_info[required_key], f_type):
            raise ValueError('Key `{}` in project info file should be of type {}'.format(required_key, f_type))
    return project_info

def find_project_root():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

def find_git_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        # .git can be file (submodule) or folder (normal repo)
        if os.path.isdir(current_dir) or os.path.isfile(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])

def cache_folder(subfolder=None):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if subfolder is None:
        f_path = os.path.join(current_folder, '..', 'data', 'cache')
    else:
        f_path = os.path.join(current_folder, '..', 'data', 'cache', subfolder)
    if not os.path.isdir(f_path):
        os.makedirs(f_path)
    return os.path.abspath(f_path)

def get_cache_path(f_name, subfolder=None):
    f_path = os.path.join(cache_folder(subfolder=subfolder), f_name)
    return os.path.abspath(f_path)

def get_data_folder():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    f_path = os.path.join(current_folder, '..', 'data')
    if os.path.isdir(f_path):
        return os.path.abspath(f_path)
    else:
        raise FileNotFoundError('Folder {0} could not be found.'.format(folder_path))
