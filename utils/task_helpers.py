import logging
from collections import Counter
import os
import json
from utils.helpers import find_folder, get_cleaned_labelled_data, get_project_info, find_project_root
import sklearn.model_selection
import pandas as pd
import boto3
from utils.s3_helper import S3Helper
from utils.stats import Stats
from utils.misc import JSONEncoder


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
    # attempt to find project info file from S3
    s3_helper = S3Helper()
    project_files = s3_helper.list('other/json/projects')
    for project_file in project_files:
        if project_file.split('/')[-1].split('-')[0] == project:
            s3_helper.download_file(project_file, project_fname)
            logger.info('Successfully initialized project "{}". Find project config file under "{}".'.format(project, project_fname))
            return
    logger.error('Could not find project "{}" remotely. You can manually initialize a project with the --template option".'.format(project))


def train_test_split(question='sentiment', test_size=0.2, seed=42, name='', balanced_labels=False, all_questions=False, label_tags=[], labelled_as=None, has_label=''):
    """Splits cleaned labelled data into training and test set"""
    def _filter_for_label_balance(df):
        """Performs undersampling for overrepresanted label classes"""
        counts = Counter(df['label'])
        min_count = min(counts.values())
        _df = pd.DataFrame()
        for l in counts.keys():
            _df = pd.concat([_df, df[df['label'] == l].sample(min_count)])
        return _df
    logger = logging.getLogger(__name__)
    questions = [question]
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
            folder_path = os.path.join(find_folder('5_labels_cleaned'), 'other', has_label_flag, question)
        else:
            folder_path = os.path.join(find_folder('5_labels_cleaned'), 'splits', question)
        train, test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        for dtype, data in [['train', train], ['test', test]]:
            f_name = '{}_{}_split_{}_seed_{}{}.csv'.format(dtype, question, int(100*test_size), seed, flags)
            f_path = os.path.join(folder_path, f_name)
            data.to_csv(f_path, index=None, encoding='utf8')
            logger.info('Successfully wrote data of {:,} examples to file {}.'.format(len(data), f_path))

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
