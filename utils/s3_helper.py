import boto3
from botocore.exceptions import ClientError
import os
import glob
import logging
from tqdm import tqdm
from contextlib import contextmanager
import time
from utils.helpers import find_project_root
import joblib
import multiprocessing
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class S3Helper:
    def __init__(self, bucket_name='crowdbreaks-prd', profile='default'):
        self.bucket_name = bucket_name
        boto3.setup_default_session(profile_name=profile)
        self.client = boto3.client('s3')
        self.bucket = boto3.resource('s3').Bucket(self.bucket_name)

    def sync(self, project_name, data_type='all', last_n_days=None):
        t_start = time.time()
        self.sync_project_info(project_name)
        if data_type == 'all' or data_type == 'streaming':
            self.sync_streaming_data(project_name, last_n_days=last_n_days)
        if data_type == 'all' or data_type == 'annotation':
            self.sync_annotation_data(project_name)
        if data_type == 'all' or data_type == 'media':
            self.sync_media_data(project_name)
        t_end = time.time()
        logger.info('Finished data sync in {:.1f} min.'.format((t_end - t_start)/60))

    def sync_project_info(self, project_name):
        project_fname = os.path.join(find_project_root(), 'project_info.json')
        project_files = self.list('other/json/projects')
        for project_file in project_files:
            if project_file.split('/')[-1].split('-')[0] == project_name:
                if not os.path.isfile(project_fname):
                    # no local file present
                    self.download_file(project_file, project_fname)
                    logger.info('Successfully initialized project "{}". Find project config file under "{}".'.format(project_name, project_fname))
                elif not self.is_up_to_date(project_file, project_fname):
                    # local file present, but not up-to-date. remove old local file and download new
                    os.remove(project_fname)
                    self.download_file(project_file, project_fname)
                    logger.info('Successfully updated project config "{}".'.format(project_fname))
                else:
                    # is up-to-date
                    logger.info('Project config file is up-to-date.')
                return
        logger.error('Could not find project "{}" remotely.".'.format(project_name))

    def sync_streaming_data(self, project_name, last_n_days=10):
        logger.info('Syncing streaming data ...')
        local_tweets_dir = os.path.join(find_project_root(), 'data', '0_raw', 'streaming')
        local_files = []
        for file_type in ['jsonl', 'gz']:
            globbed = glob.glob(os.path.join(local_tweets_dir, f'*.{file_type}'))
            local_files.extend([os.path.basename(l) for l in globbed])
        if last_n_days is not None:
            logger.info('... of the last {} days...'.format(last_n_days))
            today = datetime.now()
            sync_prefixes = []
            total_files_synced = 0
            for i in range(last_n_days):
                day = (today - timedelta(days=i)).strftime('%Y/%m/%d')
                sync_prefixes.append('tweets/{}/{}/'.format(project_name, day))
            for key_prefix in sync_prefixes:
                # Sync each day separately
                _, num_files_synced = self.download_files(key_prefix, local_tweets_dir, local_files, parallel=True)
                total_files_synced += num_files_synced
            if num_files_synced == 0:
                logger.info('Nothing to sync - Everything up-to-date.')
        else:
            key_prefix = 'tweets/{}/'.format(project_name)
            remote_files, num_files_synced = self.download_files(key_prefix, local_tweets_dir, local_files, parallel=True)
            # remove old files
            num_files_removed = self.remove_old_files(remote_files=remote_files, local_files=local_files, local_dir=local_tweets_dir)
            if num_files_synced == num_files_removed == 0:
                logger.info('Everything up-to-date.')

    def sync_annotation_data(self, project_name):
        logger.info('Syncing annotation data ...')
        for mode in ['local', 'mturk', 'public', 'other']:
            logger.info('... of type {}...'.format(mode))
            mode_prefix = mode + '-batch-job' if mode in ['local', 'mturk'] else mode
            key_prefix = 'other/csv/{}/{}-results'.format(project_name, mode_prefix)
            annotation_dir = os.path.join(find_project_root(), 'data', '3_labelled', mode)
            local_files = [os.path.basename(p) for p in glob.glob(os.path.join(annotation_dir, '*.csv'))]
            remote_files, r_num_files_synced = self.download_files(key_prefix, annotation_dir, local_files)
            r_num_files_removed = self.remove_old_files(remote_files=remote_files, local_files=local_files, local_dir=annotation_dir)
            # download all uploaded batches
            if mode in ['mturk', 'local']:
                batch_dir = os.path.join(annotation_dir, 'batches')
                if not os.path.isdir(batch_dir):
                    os.makedirs(batch_dir)
                local_files = [os.path.basename(p) for p in glob.glob(os.path.join(batch_dir, '*.csv'))]
                mode_prefix = mode + '-batch-job' if mode in ['local', 'mturk'] else mode
                key_prefix = 'other/csv/{}/{}-tweets'.format(project_name, mode_prefix)
                remote_files, t_num_files_synced = self.download_files(key_prefix, batch_dir, local_files)
                t_num_files_removed = self.remove_old_files(remote_files=remote_files, local_files=local_files, local_dir=batch_dir)
                if r_num_files_synced == r_num_files_removed == t_num_files_synced == t_num_files_removed == 0:
                    logger.info('Everything up-to-date.')

    def sync_media_data(self, project_name):
        logger.info('Syncing media data ...')
        key_prefix = 'media/{}/'.format(project_name)
        local_dir = os.path.join(find_project_root(), 'data', '0_raw', 'media')
        local_files = [os.path.basename(p) for p in glob.glob(os.path.join(os.path.join(local_dir, '*')))]
        remote_files, r_num_files_synced = self.download_files(key_prefix, local_dir, local_files)
        r_num_files_removed = self.remove_old_files(remote_files=remote_files, local_files=local_files, local_dir=local_dir)
        if r_num_files_synced == r_num_files_removed == 0:
            logger.info('Everything up-to-date.')

    def download_files(self, key_prefix, download_dir, local_files, parallel=True):
        def download_file(bucket_name, key, download_dir):
            client = boto3.client('s3')
            bucket = boto3.resource('s3').Bucket(bucket_name)
            key_fname = os.path.basename(key)
            store_path = os.path.join(download_dir, key_fname)
            bucket.download_file(key, store_path)
        if parallel:
            num_cores = max(multiprocessing.cpu_count() - 1, 1)
        else:
            num_cores = 1
        # All files on S3
        all_files = self.list(key_prefix)
        # Filter by already present files
        files_to_sync = []
        all_files_basename = []
        for full_key in all_files:
            key_name = os.path.basename(full_key)
            all_files_basename.append(key_name)
            if key_name not in local_files:
                files_to_sync.append(full_key)
        if len(files_to_sync) > 0:
            if not os.path.isdir(download_dir):
                os.makedirs(download_dir)
        # Download files
        if len(files_to_sync) == 1:
            for key in tqdm(files_to_sync):
                download_file(self.bucket_name, key, download_dir)
        elif len(files_to_sync) > 1:
            # parallel download using joblib
            parallel = joblib.Parallel(n_jobs=num_cores)
            download_file_delayed = joblib.delayed(download_file)
            parallel((download_file_delayed(self.bucket_name, key, download_dir) for key in tqdm(files_to_sync)))
        return all_files_basename, len(files_to_sync)

    def is_up_to_date(self, key, local_file):
        if not os.path.isfile(local_file):
            return False
        try:
            last_modified_resp = self.client.head_object(Bucket=self.bucket_name, Key=key)
        except ClientError:
            return False
        if 'LastModified' not in last_modified_resp:
            return False
        last_modified_remote = int(last_modified_resp['LastModified'].strftime('%s'))
        last_modified_local = int(os.path.getmtime(local_file))
        return last_modified_remote < last_modified_local

    def download_file(self, key, local=''):
        if local == '':
            local = os.path.join('.', key)
        self.bucket.download_file(key, local)

    def list(self, key_prefix):
        args = {'Bucket': self.bucket_name, 'Prefix': key_prefix}
        all_files = []
        while True:
            files = self.client.list_objects_v2(**args)
            if 'Contents' not in files:
                # No files present
                return all_files
            for f in files['Contents']:
                all_files.append(f['Key'])
            try:
                args['ContinuationToken'] = files['NextContinuationToken']
            except KeyError:
                return all_files

    def remove_old_files(self, remote_files=[], local_files=[], local_dir='.'):
        files_to_be_removed = list(set(local_files) - set(remote_files))
        files_to_be_removed = [os.path.join(local_dir, p) for p in files_to_be_removed]
        num_files = len(files_to_be_removed)
        if num_files > 0:
            logger.info('Removing {} old files...'.format(num_files))
        for f in files_to_be_removed:
            os.remove(f)
        return len(files_to_be_removed)
