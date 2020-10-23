import pandas as pd
import os
from copy import copy
from collections import defaultdict, Counter
import numpy as np
import sys
from utils.helpers import find_folder, get_labelled_data, get_parsed_data
from tqdm import tqdm
import logging

class CleanLabels(object):
    """Wrapper class for functions to process/modify labelled tweets"""

    def __init__(self, df, mode='mturk', verbose=False):
        self.logger = logging.getLogger(__name__)
        self.df = df
        self.verbose = verbose
        self.mode = mode
        if self.mode == 'mturk':
            duration_key = 'totalDurationUntilMturkSubmit'
        else:
            duration_key = 'totalDurationQuestionSequence'
        self.annotator_id = 'annotator_id'
        self.df['total_time'] = self.df[self.df.full_log.notnull()].full_log.apply(lambda i: i[duration_key]/1000 if duration_key in i else np.nan)
        self.df_qs = self.df.groupby([self.annotator_id, 'tweet_id'])
        self.df_worker = self.df.groupby(self.annotator_id)
        self.df_tweet = self.df.groupby('tweet_id')
        self.outliers = pd.DataFrame()

    def clean_labels(self, selection_criterion='majority', min_labels_cutoff=3, selection_agreement=None, is_relevant=False, exclude_incorrect=False,
            contains_keywords=False, cutoff_worker_outliers=None, allow_nan=[]):
        """
        Clean labels by
        1. Apply exclude_incorrect filter (exclude annotations which were flagged as incorrect)
        2. Apply is_relevant filter (exclude tweets which were annotated as non-relevant)
        3. Apply cutoff-worker-outliers filter
        4. Getting rid of tweets with <= `min_labels_cutoff` annotations,
        5. Apply contains_keywords filter
        6. Combine annotations based on selection_criterion

        :param selection_criterion: Can be "majority" (use majority vote) or "unanimous" (only select tweets with perfect agreement), default: majority
        :param min_labels_cutoff: Discard all tweets having less than min_labels_cutoff annotations
        :param selection_agreement: Consider only tweets with a certain level of annotation agreement. If provided overwrites selection param.
        :param is_relevant: Filter tweets which have been annotated as relevant/related, default: False
        :param exclude_incorrect: Remove annotations which have been manually flagged as incorrect, default: False
        :param contains_keywords: Remove annotations of which the text does not contain keywords
        :param cutoff_worker_outliers: Remove all annotations by workers who have agreement scores below certain Z-score threshold (a reasonable value would be 2 or 3, default: None)
        """
        labels = self.df
        num_annotations_start = len(labels)
        if num_annotations_start == 0:
            self.logger.info("No annotations present.")
            return
        require_not_nan = list(set(['id', 'text', 'question_id', 'answer_id']) - set(allow_nan))
        options = []
        if self.verbose:
            self.logger.info("Processing a total of {:,} annotations...".format(num_annotations_start))
        # Remove nans
        if self.verbose:
            self.logger.info("Removing NaN values...")
        num_before = len(labels)
        labels.dropna(subset=require_not_nan, inplace=True)
        if len(labels) == 0:
            self.logger.info("All annotations were filtered out becuase of NaN values. Use allow-nan filter to allow certain fields to be NaN.")
            return
        if self.verbose:
            self.logger.info("... removed {:,} annotations with NaN values ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
        # Exclude incorrect
        if exclude_incorrect:
            self.logger.info("Apply exclude-incorrect filter...")
            options.append('exclude-incorrect')
            num_before = len(labels)
            labels = self.exclude_incorrect(labels)
            if self.verbose:
                self.logger.info("... removed by exclude-incorrect: {:,} ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
            if len(labels) == 0:
                self.logger.info("All annotations were filtered out. No files were written.")
                return
        # Is relevant
        if is_relevant:
            self.logger.info("Apply is-relevant filter...")
            options.append('is-relevant')
            num_before = len(labels)
            labels = self.is_relevant(labels)
            if self.verbose:
                self.logger.info("... removed by is-relevant: {:,} ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
            if len(labels) == 0:
                self.logger.info("All annotations were filtered out. No files were written.")
                return
        # Cutoff worker outliers
        if cutoff_worker_outliers is not None:
            self.logger.info("Apply cutoff-worker-outliers filter...")
            options.append('cutoff-worker-outliers-{}'.format(cutoff_worker_outliers))
            num_before = len(labels)
            labels = self.cutoff_worker_outliers(labels, cutoff_worker_outliers)
            if self.verbose:
                self.logger.info("... removed by min-labels-cutoff: {:,} ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
            if len(labels) == 0:
                self.logger.info("All annotations were filtered out. No files were written.")
                return
        # Min labels cutoff
        if min_labels_cutoff > 0:
            self.logger.info("Apply min-labels-cutoff filter...")
            options.append('min-labels-cutoff-{}'.format(min_labels_cutoff))
            num_before = len(labels)
            labels = self.labels_cutoff(labels, min_labels_cutoff)
            if self.verbose:
                self.logger.info("... removed by min-labels-cutoff: {:,} ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
            if len(labels) == 0:
                self.logger.info("All annotations were filtered out. No files were written.")
                return
        if contains_keywords:
            self.logger.info("Apply contains-keywords filter...")
            options.append('contains-keywords')
            num_before = len(labels)
            labels = self.contains_keywords(labels)
            if self.verbose:
                self.logger.info("... removed by contains-keywords: {:,} ({:.2f}%)".format(num_before - len(labels), 100*(num_before - len(labels))/num_before))
            if len(labels) == 0:
                self.logger.info("All annotations were filtered out. No files were written.")
                return
        # Merge labels
        self.logger.info("Merge labels...")
        if selection_agreement is None:
            options.append(selection_criterion)
        else:
            options.append('selection-agreement-{}'.format(selection_agreement))
        labels = self.merge_labels(labels, selection_criterion, selection_agreement)
        if self.verbose:
            self.logger.info('Successfully reduced {:,} annotations into {:,} labels... '.format(num_annotations_start, len(labels)))
        # Write output files
        self.logger.info('Writing output files...')
        self.write_cleaned_labels(labels, options)

    def cutoff_worker_outliers(self, labels, cutoff_worker_outliers, min_comparisons_count=20):
        worker_agreement = self.annotator_agreement()
        worker_agreement = worker_agreement[worker_agreement['total_count'] >= min_comparisons_count]
        workers = worker_agreement.index
        # calculate agreement ratio
        worker_agreement['agreement_values'] = worker_agreement['agreement_count'] / worker_agreement['total_count']
        worker_agreement['agreement_zscores'] = self.z_scores(worker_agreement['agreement_values'], consider_only_below_median=True)
        worker_agreement['agreement_outlier'] = worker_agreement['agreement_zscores'] > cutoff_worker_outliers
        outliers = worker_agreement[worker_agreement.agreement_outlier].index
        self.logger.info('Removing annotations by {:,} annotators...'.format(len(outliers)))
        labels = labels[~labels.annotator_id.isin(outliers)]
        return labels

    def contains_keywords(self, labels):
        df = get_parsed_data(usecols=['id', 'contains_keywords'])
        labels = pd.merge(labels, df, left_on='tweet_id', right_on='id', how='inner')
        return labels[labels.contains_keywords]

    def is_relevant(self, labels):
        answer_tags = labels['answer_tag'].unique()
        relevant_tags = ['relevant', 'related', 'is_related', 'is_relevant']
        found_relevant_tag = None
        for relevant_tag in relevant_tags:
            if relevant_tag in answer_tags:
                if found_relevant_tag is not None:
                    raise ValueError('Labelled data contains multiple versions of the tags {}. Therefore, filtering for relevant-only data is ambiguous.'.format(','.join(relevant_tags)))
                else:
                    found_relevant_tag = relevant_tag
        if found_relevant_tag is None:
            raise ValueError('Labelled data does not contain a answer_tag with any of the tags {}. Therefore, filtering for relevant-only data is not possible.'.format(','.join(relevant_tags)))
        relevant_ids = labels[labels['answer_tag'] == found_relevant_tag]['tweet_id']
        return labels.loc[labels['tweet_id'].isin(relevant_ids)]

    def exclude_incorrect(self, labels):
        for i, row in tqdm(labels[labels.flag == 'incorrect'].iterrows(), desc='Excluding incorrect', total=len(labels[labels.flag == 'incorrect'])):
            identical = labels[(labels['tweet_id'] == row['tweet_id']) &
                    (labels['question_tag'] == row['question_tag']) &
                    (labels['answer_tag'] == row['answer_tag'])]
            if len(identical) > 0:
                # set identical annotations also to incorrect
                labels.loc[identical.index, 'flag'] = 'incorrect'
        labels = labels[labels.flag.isin(['default', 'correct'])]
        return labels

    def labels_cutoff(self, labels, min_labels_cutoff):
        labels = labels.groupby(['tweet_id', 'question_tag'])
        labels = labels.filter(lambda x: x['question_tag'].count() >= min_labels_cutoff)
        return labels

    def merge_labels(self, labels, selection_criterion, selection_agreement):
        labels_merged = []
        def _get_answer_tag(group):
            answers = Counter(g['answer_tag'])
            num_answers = len(g)
            if selection_agreement is not None:
                candidates = [k for k, v in answers.items() if v == max(answers.values())]
                answer_tag = candidates[0];
                if answers[answer_tag] / len(g) > selection_agreement:
                    return answer_tag
            else:
                if selection_criterion == 'majority':
                    candidates = [k for k, v in answers.items() if v == max(answers.values())]
                    # only consider cases when there is no tie
                    if len(candidates) == 1:
                        return candidates[0]
                elif selection_criterion == 'unanimous':
                    if len(answers) == 1:
                        return list(answers.keys())[0]
        for (tweet_id, question_tag), g in tqdm(labels.groupby(['tweet_id', 'question_tag']), desc='Merging'):
            answer_tag =  _get_answer_tag(g)
            if answer_tag is not None:
                labels_merged.append({
                    'text': g['text'].values[0],
                    'question_id': g['question_id'].values[0],
                    'question_tag': question_tag,
                    'label_id': g['answer_id'].values[0],
                    'label_tag': answer_tag,
                    'id': g['tweet_id'].values[0]
                    })
        return pd.DataFrame(labels_merged)

    def write_cleaned_labels(self, labels, options):
        def _create_dir(dir_name):
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
        if len(labels) == 0:
            self.logger.info('No labels to write. Maybe the filtering parameters are too strict. Aborting.')
            return
        # write full file
        folder_path = find_folder('4_labels_cleaned')
        option_flags = '_'.join(options)
        f_path = os.path.join(folder_path, 'cleaned_labels_{}.csv'.format(option_flags))
        labels[['id', 'text', 'question_id', 'question_tag', 'label_id', 'label_tag']].to_csv(f_path, index=False)
        self.logger.info('Successfully wrote {:,} labels to file {}'.format(len(labels), f_path))
        # write 1 file per question
        folder_path_by_question = os.path.join(folder_path, 'by_question')
        labels.rename(columns={'label_tag': 'label'}, inplace=True)
        self.logger.info('Writing one files by question...')
        for question_tag, g in labels.groupby('question_tag'):
            f_path = os.path.join(folder_path_by_question, question_tag, 'cleaned_labels_{}_{}.csv'.format(option_flags, question_tag))
            _create_dir(os.path.dirname(f_path))
            g[['id', 'text', 'label']].to_csv(f_path, index=False)
            self.logger.info('... successfully wrote {:,} labels for question {} to file {}'.format(len(g), question_tag, f_path))

    # QS-level analysis
    def plt_number_results_per_qs(self):
        self.plot_counter_hist(Counter(self.df_qs.count()['id']), title='Number of results per question sequence')

    def plt_av_time_qs(self, log=False, min_n=None, title='Average time per question sequence (s)', subtitle=''):
        av_times = self.df_qs['total_time'].first().values
        if min_n is not None:
            av_times = sorted(av_times)[:min_n]
        if not subtitle == '':
            title = title + '\n' + subtitle
        ax = self.plot_hist(av_times, title=title, xlog=log, xlabel='Time to submit (s)', x_origin=True)
        if min_n is None:
            mean = np.mean(av_times)
            median = np.median(av_times)
            ax.axvline(x=mean, color='k', lw=0.5, alpha=0.6)
            ax.axvline(x=median, color='k', lw=0.5, alpha=0.6)
            y_pos = ax.get_ylim()[1] * 0.8
            ax.text(mean*1.1, y_pos, 'Mean = {:.2f}s'.format(mean))
            ax.text(median*1.1, y_pos * 0.9, 'Median = {:.2f}s'.format(median))

    # Worker-level analysis
    def plt_number_results_per_worker(self):
        self.plot_hist(self.df_worker.count()['id'], title='Result distribution by worker', xlabel='Number of results', ylabel='Number of workers')

    def plt_qs_per_worker(self):
        self.plot_hist(self.qs.groupby(self.annotator_id).count()['qs_count'], title='Number of QS per worker', xlabel='Number of QS', ylabel='Counts')

    def plt_mean_time_per_worker(self, min_n=None, title='Mean time (s) by worker', subtitle=''):
        av_times = self.df_worker['total_time'].mean()
        if min_n is not None:
            av_times = sorted(av_times)[:min_n]
        if not subtitle == '':
            title = title + '\n' + subtitle
        self.plot_hist(av_times, title=title, xlabel='Time (s)', ylabel='Number of workers')

    def compute_fleiss_kappa(self, question_tag):
        """Compute Fleiss' Kappa for a question."""
        # compute table (Nxk), N being the number of tweets, k being the number of possible answers for this question
        df_question = self.df[self.df['question_tag'] == question_tag]
        N = len(df_question['tweet_id'].unique())
        k = len(df_question['answer_tag'].unique())
        table = pd.DataFrame(0, index=df_question['tweet_id'].unique(), columns=df_question['answer_tag'].unique())
        for i, row in df_question.iterrows():
            table.loc[row.tweet_id][row.answer_tag] += 1
        # number of annotations given by tweet
        num_annotations = table.sum(axis=1)
        # num agreements (how many annotator-annotator pairs are in agreement): sum(n_ij * (n_ij - 1)) = sum(n_ij ** 2) - sum(n_ij)
        num_agreements = table.pow(2).sum(axis=1) - num_annotations
        # num possible agreements (how many possible annotator-annotator pairs there are)
        num_possible_agreements = num_annotations * (num_annotations - 1)
        # agreement
        table['agreement'] = num_agreements/num_possible_agreements
        # compute chance agreement
        num_annotations = table.sum().sum()
        answer_fractions = table.sum(axis=0)/num_annotations
        chance_agreeement = answer_fractions.pow(2).sum()
        # Fleiss' Kappa: (Mean observed agreement - chance agreement)/(1 - chance agreement)
        fleiss_kappa = (table['agreement'].mean() - chance_agreeement)/(1 - chance_agreeement)
        return fleiss_kappa

    def annotator_agreement(self):
        agreement = defaultdict(lambda: {'agreement_count': 0, 'total_count': 0})
        for tweet_id, g in tqdm(self.df_tweet, desc='Computing annotator agreement', total=len(self.df_tweet)):
            mappings = {}
            for worker_id, tweet_worker in g.groupby(self.annotator_id):
                mappings[worker_id] = dict(zip(tweet_worker['question_tag'], tweet_worker['answer_tag']))
            df_mappings = pd.DataFrame(mappings)
            for worker_id1 in df_mappings:
                for worker_id2 in df_mappings.loc[:, df_mappings.columns != worker_id1]:
                    cleaned_df_mappings = df_mappings[[worker_id1, worker_id2]].dropna()
                    num_agreements = np.sum(cleaned_df_mappings[worker_id1] == cleaned_df_mappings[worker_id2])
                    count = len(cleaned_df_mappings)
                    agreement[worker_id1]['agreement_count'] += num_agreements
                    agreement[worker_id1]['total_count'] += count
        agreement = pd.DataFrame(agreement).T
        return agreement

    def worker_outliers(self, min_tasks=3, time_cutoff=3, agreement_cutoff=3, min_comparisons_count=20, plt=True, print_summary=True, compute_only=None):
        """Detect workers which performed significantly worse than others. Returns dict with worker as keys and outlier tags
        :param min_tasks: Only consider workers with at least min_tasks submissions.
        :param time_cutoff: Z-score cut-off to be used for the mean time to completion of task.
        :param min_comparisons_count: Minimum number of comparisons between workers so that agreement is calculated. If None ignored.
        :param plt: Outlier plots
        :param print_summary: Print list of outlier workers
        """
        qs_counts_by_worker = self.qs.groupby(self.annotator_id).count()['qs_count']
        self.outliers = pd.DataFrame(index=qs_counts_by_worker.index)
        # Mean time per worker outliers
        workers = qs_counts_by_worker[qs_counts_by_worker >= min_tasks].index
        self.outliers.loc[workers, 'time_values'] = self.df_worker['total_time'].mean()[workers]
        self.outliers.loc[workers, 'time_zscores'] = self.z_scores(self.outliers.loc[workers, 'time_values'], consider_only_below_median=True)
        self.outliers.loc[workers, 'time_outlier'] = ['outlier' if i > time_cutoff else 'no_outlier' for i in self.outliers.loc[workers, 'time_zscores']]
        # Mean agreement per worker
        worker_agreement = self.annotator_agreement()
        if min_comparisons_count is not None:
            worker_agreement = worker_agreement[worker_agreement['total_count'] >= min_comparisons_count]
        workers = worker_agreement.index
        # calculate agreement ratio
        self.outliers.loc[workers, 'agreement_values'] = worker_agreement['agreement_count'] / worker_agreement['total_count']
        self.outliers['agreement_zscores'] = self.z_scores(self.outliers.loc[workers, 'agreement_values'], consider_only_below_median=True)
        self.outliers.loc[workers, 'agreement_outlier'] = self.outliers.loc[workers, 'agreement_zscores'] > agreement_cutoff
        self.outliers.loc[workers, 'agreement_outlier'] = ['outlier' if i > agreement_cutoff else 'no_outlier' for i in self.outliers.loc[workers, 'agreement_zscores']]
        if plt:
            self.plt_outliers()
        return self.outliers

    def z_scores(self, x, consider_only_below_median=False):
        """
        Calculate z-scores based on deviation from the median (multiples of MAD (median absolute deviation), a measure which is more robust and less affected by outliers)
        """
        x_median = np.median(x)
        mad = np.median(np.abs(x - x_median))
        if consider_only_below_median:
            x = x[x < x_median]
        return np.abs(x - x_median)/mad

    def plt_outliers(self):
        import seaborn as sb
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        dotsize = 2
        _ = sb.swarmplot(x='time_values', y=[""]*len(self.outliers), data=self.outliers, hue=self.outliers['time_outlier'], ax=ax[0], size=dotsize)
        _ = sb.swarmplot(x='agreement_values', y=[""]*len(self.outliers), data=self.outliers, hue=self.outliers['agreement_outlier'], ax=ax[1], size=dotsize)
        for _ax in ax:
            _ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
            xlims = _ax.get_xlim()
            offset = (xlims[1] - xlims[0])*0.1
            _ax.set_xlim([xlims[0]-offset, xlims[1]+offset])
        plt.subplots_adjust(hspace=1)
        # plt.tight_layout()

    # Tweet-level analysis
    @property
    def qs(self):
        qs = self.df_qs.size().reset_index()
        qs['qs_count'] = 1
        qs.rename({0: 'results_count'}, axis='columns', inplace=True)
        return qs

    def plt_number_qs_by_tweet(self):
        self.plot_counter_hist(Counter(self.qs.groupby('tweet_id').count()['qs_count'].values), title='Number of QS by tweet', xlabel='Number of QS', ylabel='Counts')

    def plt_number_unique_workers_per_tweet(self):
        tweet_worker_map = defaultdict(set)
        for g, i in self.df_qs:
            tweet_worker_map[g[1]].add(g[0])
        unique_counts = [len(i) for i in tweet_worker_map.values()]
        self.plot_counter_hist(Counter(unique_counts), title='Number of unique workers by tweet', xlabel='Number of unique workers', ylabel='Counts')

    def plot_counter_hist(self, counter_dict, title='', sort=True, sort_by='key', xlabel='', ylabel='Counts'):
        if sort:
            if sort_by == 'key':
                counter_dict = dict(sorted(counter_dict.items(), key=lambda kv: int(kv[0])))
            else:
                counter_dict = dict(sorted(counter_dict.items(), key=lambda kv: int(kv[1])))
        labels, values = zip(*counter_dict.items())
        indexes = np.arange(len(labels))
        width = .5
        plt.bar(indexes, values, width)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(indexes, labels)
        ax = plt.gca()
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[0], ylims[1]*1.1)
        return ax

    def plot_hist(self, x, title='', xlabel='', ylabel='counts', xlog=False, x_origin=False, orientation='vertical', bins='auto'):
        plt.hist(x, bins=bins, orientation=orientation)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlog:
            plt.xscale('log')
        ax = plt.gca()
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[0], ylims[1]*1.1)
        if x_origin and not xlog:
            ax.set_xlim(0, None)
        return ax

def run_clean_labels(selection_criterion='majority', min_labels_cutoff=3, selection_agreement=None, mode='*', is_relevant=False, exclude_incorrect=False,
        cutoff_worker_outliers=None, allow_nan=[], contains_keywords=False, verbose=False):
    logger = logging.getLogger(__name__)
    logger.info('Reading labelled data...')
    df = get_labelled_data(mode=mode)
    pl = CleanLabels(df, mode=mode, verbose=verbose)
    pl.clean_labels(selection_criterion=selection_criterion, min_labels_cutoff=min_labels_cutoff,
            selection_agreement=selection_agreement, is_relevant=is_relevant, exclude_incorrect=exclude_incorrect,
            cutoff_worker_outliers=cutoff_worker_outliers, allow_nan=allow_nan, contains_keywords=contains_keywords)
