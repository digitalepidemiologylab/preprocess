import pandas as pd
import os
import re
import numpy as np
import io
import glob
import json
import warnings
import csv
import logging
import re


def get_merged_data(dtype='original', year=None, frac=1.0, usecols=None):
    """Read raw data
    :param dtype: possible values: "original", "anonymized", "encrypted"
    :param year: Read data of a certain year
    :param frac: fraction of data to be read (default 1)
    :param usecols: List of columns to read from
    """
    f_name = get_f_name(dtype, year, processing_step='merged')
    f_path = find_file(f_name, subfolder='1_merged')
    return read_raw_data_from_csv(f_path, dtype, frac=frac, usecols=usecols)

def get_cleaned_data(dtype='original', year=None, frac=1.0, contains_keywords=False, flag=None, usecols=None):
    """Read cleaned data
    :param dtype: possible values: "original", "anonymized", "encrypted"
    :param year: Year is not used anymore
    :param frac: fraction of data to be read (default 1)
    :param contains_keywords: Only get data where contains_keywords is True (default: False)
    :param flag: Can be None, 'use_for_prediction', 'use_for_labelling'. None gives all data, use_for_prediction/labelling gives corresponding
    labelled subset. The first time used, the subset will be cached for subsequent use.
    :param usecols: Only extract certain columns (default: all columns)
    """
    f_name = get_f_name(dtype, year, processing_step='cleaned', flag=flag, contains_keywords=contains_keywords)
    if flag is not None or contains_keywords:
        f_path = find_file(f_name, subfolder='2_cleaned', cached=True)
        if f_path == '':
            # No cached file found, read full file
            f_name_full = get_f_name(dtype, year, processing_step='cleaned')
            f_path_full = find_file(f_name_full, subfolder='2_cleaned')
            df = read_raw_data_from_csv(f_path_full, dtype, frac=frac, usecols=usecols)
            if flag is not None:
                df = df[df[flag]]
            if contains_keywords:
                df = df[df['contains_keywords']]
            # write cached file
            path = os.path.join(os.path.dirname(f_path_full), f_name)
            df.to_csv(path, encoding='utf8')
            return df
    else:
        f_path = find_file(f_name, subfolder='2_cleaned')
    return read_raw_data_from_csv(f_path, dtype, frac=frac, usecols=usecols)

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
    f_names = glob.glob(os.path.join(find_folder('4_labelled'), mode, '*{}*.csv'.format(pattern)))
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
        logger = logging.getLogger(__name__)
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
    logger = logging.getLogger(__name__)
    if name == '':
        f_path = os.path.join(find_folder('5_labels_cleaned'), 'cleaned_labels*.csv')
        label_files = glob.glob(f_path)
        if len(label_files) == 1:
            f_path = label_files[0]
        elif len(label_files) == 0:
            raise FileNotFoundError('No cleaned label files could be found with the pattern {}'.format(f_path))
        else:
            raise ValueError('Found {} different files for cleaned labels. Provide "name" argument to specify which.'.format(len(label_files)))
    else:
        f_path = os.path.join(find_folder('5_labels_cleaned'), '{}.csv'.format(name))
    dtypes = {'id': str, 'question_id': 'Int64', 'answer_id': 'Int64'}
    df = pd.read_csv(f_path, encoding='utf8', dtype=dtypes)
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
    f_names = glob.glob(os.path.join(find_folder('3_sampled'), 'sampled_{}_*.csv'.format(dtype)))
    logger = logging.getLogger(__name__)
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
    f_names = glob.glob(os.path.join(find_folder('3_sampled'), 'batch_*', 'batch_*.csv'))
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
    f_names = glob.glob(os.path.join(find_folder('4_labelled'), mode, 'batches', '*.csv'))
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
    f_names = glob.glob(os.path.join(find_folder('4_labelled'), mode, 'batches', '*.csv'))
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
    Return prediction data. Prediction data should be in `data/6_predicted`. All file names should have the following pattern: `predicted_{column_name}_{YYY}-{MM}-{dd}_{5-char-hash}.csv`.
    The column name can be an arbitrary tag (e.g. question tag) which should be unique.
    :param include_raw_data: Merge predictions with 2_cleaned_data (by default True)
    :param dtype: Raw data dtype (default: anonymized), relevant if include_raw_data is set to True.
    :param flag: Filter raw data by certain flags, relevant if include_raw_data is set to True.
    :param drop_retweets: Drop retweets (default: False), relevant if include_raw_data is set to True.
    :param usecols: Only select certains columns from 2_cleaned_data (by default all), relevant if include_raw_data is set to True.
    """
    f_pattern = os.path.join(find_folder('6_predicted'), 'predicted_*.csv')
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
        df = get_cleaned_data(dtype=dtype, usecols=usecols)
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

def get_all_data(include_all_data=True, dtype='anonymized', token_count_cutoff=3, s_date='2013-01-01', e_date='2019-05-31', mode='mturk',
        cleaned_labels_name_1='cleaned_labels_cutoff-worker-outliers-3.0_min-labels-cutoff-3_unanimous',
        cleaned_labels_name_2='cleaned_labels_is-relevant_cutoff-worker-outliers-3.0_min-labels-cutoff-3_unanimous',
        include_predictions=True):
    """
    Returns all data including predictions and filters
    :param include_all_data: If set to False return the minimal possible number of columns (id, predictions, filters), default: True
    :param dtype: Raw data dtype (default: anonymized)
    :param s_date: Start date filter
    :param e_date: End date filter
    :param token_count_cutoff: Word token cut-off (default: 3)
    :param cleaned_labels_name_1: Cleaned label filter 1
    :param cleaned_labels_name_2: Cleaned label filter 2
    """
    # load data
    if include_all_data:
        usecols = None
    else:
        usecols = ['id', 'is_duplicate', 'token_count', 'extracted_quoted_tweet', 'is_retweet', 'contains_keywords', 'created_at']
    df = get_cleaned_data(dtype=dtype, usecols=usecols)
    df_sampled = get_uploaded_batched_data(mode=mode)
    df_available = get_uploaded_batched_data(availability='available', mode=mode)
    df_labelled = get_labelled_data(usecols=['tweet_id'], mode=mode)
    df_cleaned_labels_1 = get_cleaned_labelled_data(name=cleaned_labels_name_1, cols=['id'])
    df_cleaned_labels_2 = get_cleaned_labelled_data(name=cleaned_labels_name_2, cols=['id'])
    if include_predictions:
        df_pred = get_predicted_data(include_raw_data=False, dtype=dtype)
        df_pred.index = df.index
        df = pd.concat([df, df_pred], axis=1)
    # compute filters for raw data
    df['d0'] = (df.contains_keywords) & (df.index >= s_date) & (df.index <= e_date)
    df['d1'] = (df.d0) & (df.token_count >= token_count_cutoff)
    df['d2'] = (df.d1) & (~df.is_retweet) & (~df.is_duplicate) & (~df.extracted_quoted_tweet)
    # compute filters for sampling data
    df['s0'] = df.id.isin(df_sampled.tweet_id)
    df['s1'] = (df.s0) & (df.id.isin(set(df_available['tweet_id'])))
    # compute filters for annotation data
    df['a0'] = df.id.isin(df_labelled.tweet_id)
    df['a1'] = df.id.isin(df_cleaned_labels_1.id)
    df['a2'] = df.id.isin(df_cleaned_labels_2.id)
    # compute filters for prediction data
    if include_predictions:
        df['p0'] = df.d1
        df['p1'] = (df.p0) & (df.label_related == 'related')
    return df

def read_raw_data_from_csv(f_path, dtype, frac=1, usecols=None, parallel=False, set_index='created_at'):
    def env_is_true(env_var):
        env_var = os.environ.get(env_var, '')
        return env_var == '1' or env_var.lower() == 'true'
    dtypes = {
            'created_at': str,
            'contains_keywords': bool,
            'entities.hashtags': str,
            'entities.user_mentions': str,
            'extracted_quoted_tweet': bool,
            'favorite_count': 'Int64',
            'has_media': bool,
            'has_quoted_status': bool,
            'id': str,
            'in_reply_to_status_id': str,
            'in_reply_to_user_id': str,
            'is_retweet': bool,
            'latitude': 'float',
            'longitude': 'float',
            'media': str,
            'place.bounding_box': str,
            'place.country_code': str,
            'place.full_name': str,
            'place.place_type': str,
            'quoted_status.favorite_count': 'Int64',
            'quoted_status.has_media': str,
            'quoted_status.id': str,
            'quoted_status.in_reply_to_status_id': str,
            'quoted_status.media': str,
            'quoted_status.retweet_count': 'float',
            'quoted_status.text': str,
            'quoted_status.user.followers_count': 'Int64',
            'quoted_status.user.id': str,
            'reply_count': 'Int64',
            'retweet_count': 'Int64',
            'retweeted_status.favorite_count': 'Int64',
            'retweeted_status.id': str,
            'retweeted_status.in_reply_to_status_id': str,
            'retweeted_status.retweet_count': 'Int64',
            'retweeted_status.user.followers_count': 'Int64',
            'retweeted_status.user.id': str,
            'text': str,
            'user.followers_count': 'Int64',
            'user.friends_count': 'Int64',
            'user.id': str,
            'user.location': str,
            'user.name': str,
            'user.screen_name': str,
            'is_duplicate': bool,
            'token_count': 'Int64',
            'use_for_labelling': bool,
            'use_for_prediction': bool
            }
    if usecols is not None:
        dtypes = {i:v for i,v in dtypes.items() if i in usecols}
    if parallel or env_is_true('PARALLEL'):
        try:
            import modin.pandas as pd
        except:
            # Fallback to normal pandas
            import pandas as pd
    else:
        import pandas as pd
    if not (isinstance(frac, int) or isinstance(frac, float)):
        raise ValueError('Frac has to be of type integer or float')
    if frac < 0 or frac > 1:
        raise ValueError('Frac should be between 0 and 1')
    if int(frac) == 1:
        df = pd.read_csv(f_path, encoding='utf8', dtype=dtypes, usecols=usecols)
    else:
        # read only fraction of csv
        with open(f_path, 'r') as csvfile:
            num_rows = sum(1 for row in csvfile)
            csvfile.seek(0)
            break_at = int(frac * num_rows)
            if break_at == 0: return pd.DataFrame()
            df = ''
            for i, line in enumerate(csvfile):
                df += line
                if i >= break_at:
                    break
        df = pd.read_csv(io.StringIO(df), dtype=dtypes, encoding='utf8', usecols=usecols)
    if len(df) <= 1:
        return df
    if 'created_at' in df:
        df['created_at'] = pd.to_datetime(df['created_at'].values, utc=True)
    if set_index is not None and set_index in df:
        df.set_index(set_index, drop=True, inplace=True)
    return df

def get_f_name(dtype, year, processing_step='merged', flag=None, contains_keywords=False):
    if flag is None:
        flag = ''
    else:
        flag = '_' + flag
    if contains_keywords:
        contains_keywords_flag = '_contains_keywords'
    else:
        contains_keywords_flag = ''
    if year is None:
        f_name = '{}_{}{}{}.csv'.format(processing_step, dtype, flag, contains_keywords_flag)
    else:
        if not (isinstance(year, str) or isinstance(year, int)):
            raise ValueError('Please provide the year as an integer or string')
        f_name = '{}_{}_year_{}{}{}.csv'.format(processing_step, dtype, year, flag, contains_keywords_flag)
    return f_name

def find_file(f_name, subfolder='1_merged', num_parent_dirs=4, cached=False):
    data_folder = 'data'
    for i in range(num_parent_dirs):
        par_dirs = i*['..']
        f_path = os.path.join(*par_dirs, data_folder, subfolder, f_name)
        if os.path.isfile(f_path):
            return f_path
    if not cached:
        raise FileNotFoundError('File {0} could not be found.'.format(f_name))
    else:
        return ''

def find_folder(folder_path, subfolder='data', num_par_dirs=4):
    if subfolder == '.':
        # ignore subfolder if current folder
        subfolder = ''
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        f_path = os.path.join('.', *par_dirs, subfolder, folder_path)
        if os.path.isdir(f_path):
            return f_path
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

def find_project_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])
