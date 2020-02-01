from utils.helpers import get_project_info, find_folder, get_labelled_data, get_sampled_data, get_cleaned_labelled_data, find_project_root, get_batched_sample_data, get_uploaded_batched_data
from utils.processing.clean_labels import CleanLabels
import os
import logging
import glob
import numpy as np
import pandas as pd
import subprocess

class Stats(object):
    """Outputs general statistcs for current project"""

    def __init__(self):
        self.text = ''
        self.logger = logging.getLogger(__name__)
        self.filler = 40
        self.project_header()

    def all(self):
        self.overview()
        self.separator()
        self.sample()
        self.separator()
        self.annotation()
        self.separator()
        self.annotator_outliers()
        self.separator()
        self.annotation_cleaned()
        self.write_to_file()
        for fmt in ['html', 'pdf']:
            self.convert(to_fmt=fmt)

    def overview(self):
        self.header('Overview')
        self.raw_data()
        self.parsed_data()
        self.sampled_data()
        self.annotation_data()
        self.cleaned_annotation_data()

    def sample(self):
        self.header('Sample')
        # load data
        try:
            df_samples = get_sampled_data()
        except FileNotFoundError:
            self.text += 'No sampling file(s) present.\n'
            return
        df_batched = get_batched_sample_data()
        tweet_ids_batched = set(df_batched['tweet_id'])
        df_unavailable = get_uploaded_batched_data(availability='unavailable')
        tweet_ids_unavailable = set(df_unavailable['tweet_id'])
        tweet_ids_sampled = set(df_samples['tweet_id'])
        still_available = tweet_ids_sampled - tweet_ids_unavailable - tweet_ids_batched
        self.add_key_value('Tweets in sample', len(tweet_ids_sampled))
        self.add_key_value('- unavailable', len(tweet_ids_unavailable))
        self.add_key_value('- in batches', len(tweet_ids_batched))
        self.add_key_value('- left to sample', len(still_available))
        self.text += '\n'

    def annotation(self, mode='all', pattern='*', min_annotations=20):
        self.header('Annotation')
        if mode == 'all':
            mode = ['public', 'local', 'mturk', 'other']
        else:
            mode = [mode]
        for _mode in mode:
            self.make_title(_mode.capitalize())
            try:
                self.logger.info('Reading {} annotation data...'.format(_mode))
                df = get_labelled_data(pattern=pattern, mode=_mode)
            except FileNotFoundError:
                self.text += 'No {} annotations present.\n\n'.format(_mode)
                continue
            # only select questions for which there have been min_annotations annotations
            question_tags = []
            for question_tag, df_question in df.groupby('question_tag'):
                if len(df_question.tweet_id.unique()) > min_annotations:
                    question_tags.append(question_tag)
            self.logger.info('Compute agreement summary for {} annotations...'.format(_mode))
            pl = CleanLabels(df, mode=_mode)
            self.text += "\nAgreement (Fleiss' kappa) by question:\n"
            for question_tag in question_tags:
                fleiss_kappa = pl.compute_fleiss_kappa(question_tag)
                self.add_key_value('- {}'.format(question_tag), fleiss_kappa, fmt='.2f')
            self.text += '\n'

    def annotator_outliers(self, mode='mturk', batch_name='*', time_cutoff=3, agreement_cutoff=3, min_tasks=20, min_comparisons_count=20):
        self.header('Annotator outliers')
        try:
            self.logger.info('Reading annotation data...')
            df = get_labelled_data(pattern=batch_name, mode=mode)
        except FileNotFoundError:
            self.text += 'No mturk annotations present.'
            return
        pl = CleanLabels(df, mode=mode)
        self.logger.info('Compute outliers...')
        outliers = pl.worker_outliers(time_cutoff=time_cutoff, agreement_cutoff=agreement_cutoff, plt=False, min_tasks=min_tasks, min_comparisons_count=min_comparisons_count)
        self.text += 'Outliers based on Z-score agreement cut-offs of {} (agreement) and {} (time).\n\n'.format(agreement_cutoff, time_cutoff)
        self.add_key_value('Number of workers considered', len(outliers))
        self.text += '\n'
        for by, value_fmt, factor in zip(['agreement', 'time'], [{'fmt': '.2f', 'unit': '%'}, {'fmt': '.1f', 'unit': 's'}], [100, 1]):
            self.make_title('By {}'.format(by))
            outlier_key = '{}_outlier'.format(by)
            value_key = '{}_values'.format(by)
            num_outliers = np.array(outliers[outlier_key] == 'outlier').sum()
            num_no_outliers = np.array(outliers[outlier_key] == 'no_outlier').sum()
            num_not_considered = outliers[outlier_key].isnull().sum()
            self.add_key_value('- Outliers', num_outliers, with_percent=100*num_outliers/len(outliers))
            self.add_key_value('- Not outliers', num_no_outliers, with_percent=100*num_no_outliers/len(outliers))
            self.add_key_value('- Not considered', num_not_considered, with_percent=100*num_not_considered/len(outliers))
            self.text +=  '\n'
            self.add_key_value('- Mean {}'.format(by), factor*outliers[value_key].mean(), **value_fmt)
            self.add_key_value('- Median {}'.format(by), factor*outliers[value_key].median(), **value_fmt)
            self.add_key_value('- Min {}'.format(by), factor*outliers[value_key].min(), **value_fmt)
            self.add_key_value('- Max {}'.format(by), factor*outliers[value_key].max(), **value_fmt)
            self.text += '\nOutliers for {}:\n'.format(by)
            only_outliers = outliers[outliers[outlier_key] == 'outlier'].sort_values(value_key)
            if len(only_outliers) > 0:
                for annotator_id, row in only_outliers.iterrows():
                    self.add_key_value('- {}'.format(annotator_id), factor*row[value_key], **value_fmt)
            else:
                self.text += '\nNo outliers found.\n'
            self.text += '\n'
        both_outliers = outliers[(outliers['time_outlier'] == 'outlier') & (outliers['agreement_outlier'] == 'outlier')].index.tolist()
        self.make_title('Outliers for both')
        if len(both_outliers) > 0:
            self.text += '\n'.join(both_outliers)
        else:
            self.text += '\nNo outliers found.\n'
        self.text += '\n'

    def annotation_cleaned(self):
        self.header('Cleaned annotations')
        f_path = os.path.join(find_folder('4_labels_cleaned'), 'cleaned_labels.csv')
        try:
            self.logger.info('Reading cleaned annotation data...')
            df = pd.read_csv(f_path)
        except FileNotFoundError:
            self.text += 'No cleaned annotations present.'
            return
        self.add_key_value('- Num annotation results', len(df))
        self.add_key_value('- Num tweets annotated', len(df.id.unique()))
        self.text += '\n\n'
        for question_tag, q_group in df.groupby('question_tag'):
            self.make_title('Question {}'.format(question_tag))
            total = q_group.count()['id']
            for label_tag, q_a_group in q_group.groupby('label_tag'):
                label_tag_count = q_a_group.count()['id']
                self.add_key_value('- {}'.format(label_tag), label_tag_count, with_percent=100*label_tag_count/total)
            self.text += '\n\n'

    # helpers

    def project_header(self):
        project_info = get_project_info()
        self.make_title('Stats for {}'.format(project_info['name']), title_type='project')

    def header(self, title):
        self.make_title(title, title_type='main')

    def separator(self):
        self.text += '{}\n\n'.format(70*'-')

    def raw_data(self):
        raw_data_folder = find_folder('0_raw')
        num_historic = len(glob.glob(os.path.join(raw_data_folder, 'historic', '*.json*')))
        num_streaming = len(glob.glob(os.path.join(raw_data_folder, 'streaming', '**', '*.json*')))
        total = num_historic + num_streaming
        self.make_title('Raw data')
        if total > 0:
            self.add_key_value('Number of files in raw data', total)
            self.add_key_value('- historic', num_historic)
            self.add_key_value('- streaming', num_streaming)
        else:
            self.text += 'No raw data present.\n'
        self.text += '\n'

    def parsed_data(self):
        num_lines = 0
        for dtype in ['original', 'anonymized', 'encrypted']:
            path = os.path.join(find_folder('1_parsed'), 'parsed_{}.csv'.format(dtype))
            if os.path.isfile(path):
                num_lines = sum(1 for line in open(path))
                break
        self.make_title('Parsed data')
        if num_lines > 0:
            self.add_key_value('Num tweets in parsed data', num_lines - 1)
        else:
            self.text += 'No parsed data present.\n'
        self.text += '\n'

    def sampled_data(self):
        self.make_title('Sampled data')
        f_names = glob.glob(os.path.join(find_folder('2_sampled'), 'sampled_{}_{}_*.csv'.format('*', '*')))
        num_sample_files = len(f_names)
        num_tweets_sampled = 0
        for f_name in f_names:
            num_tweets_sampled += sum(1 for line in open(f_name))
        self.add_key_value('Number of tweets sampled ({:,} file(s))'.format(num_sample_files), num_tweets_sampled)
        self.text += '\n'

    def annotation_data(self):
        self.make_title('Annotation data')
        self.text += 'Number of annotation results:\n'
        for mode in ['public', 'local', 'mturk', 'other']:
            f_names = glob.glob(os.path.join(find_folder('3_labelled'), mode, '*.csv'))
            num_annotations = 0
            for f_name in f_names:
                num_annotations += sum(1 for line in open(f_name))
            self.add_key_value('- {}'.format(mode), num_annotations)
        self.text += '\n'

    def cleaned_annotation_data(self):
        self.make_title('Cleaned annotation data')
        f_name = os.path.join(find_folder('4_labels_cleaned'), 'cleaned_labels.csv')
        num_annotations = 0
        if os.path.isfile(f_name):
            num_annotations = sum(1 for line in open(f_name))
        self.add_key_value('Number of cleaned annotations', num_annotations)
        self.text += '\n'

    def make_title(self, title, title_type=None):
        if title_type == 'project':
            level = 1
        elif title_type == 'main':
            level = 2
        else:
            level = 3
        self.text += '{} {}\n\n'.format(level*'#', title)

    def add_key_value(self, key, value, fmt=',', width=12, unit='', with_percent=None):
        if with_percent:
            self.text += '{}:{}{:>{width}{fmt}}{unit} ({with_percent:.2f}%)\n'.format(key, max(0, self.filler - len(key))*' ', value, width=width, fmt=fmt, unit=' '+unit, with_percent=with_percent)
        else:
            self.text += '{}:{}{:>{width}{fmt}}{unit}\n'.format(key, max(0, self.filler - len(key))*' ', value, width=width, fmt=fmt, unit=' '+unit)

    def write_to_file(self, f_name='stats.md'):
        f_path = os.path.join(find_project_root(), f_name)
        self.logger.info('Writing output to file {}...'.format(f_path))
        with open(f_path, 'w') as f:
            f.write(self.text)

    def convert(self, f_name='stats.md', from_fmt='gfm', to_fmt='html'):
        """Use pandoc to convert markdown to other formats"""
        project_root = find_project_root()
        f_path = os.path.join(project_root, f_name)
        out_path = os.path.join(project_root, 'stats.' + to_fmt)
        if to_fmt == 'pdf':
            args = ['pandoc', f_path, '-f', from_fmt, '-o', out_path]
        else:
            args = ['pandoc', f_path, '-f', from_fmt, '-t', to_fmt, '-o', out_path]
        try:
            subprocess.check_call(args)
        except:
            self.logger.error('Conversion not successful. Make sure pandoc is installed on your system.')




