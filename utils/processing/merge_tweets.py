import math
import re
import pandas as pd
import os
from copy import copy
from collections import defaultdict
from html.parser import HTMLParser
from utils.processing.encrypt import Encrypt
import glob
import re
import ast
import time
import json
import unicodedata
from tqdm import tqdm
from munch import DefaultMunch
import multiprocessing
import joblib
import itertools
import warnings
import utils.helpers
import logging
import shapely.geometry

class MergeTweet(object):
    """Wrapper class for functions to process/modify tweets"""

    def __init__(self, tweet=None):
        self.tweet = tweet
        self.extended_tweet = self._get_extended_tweet()
        self.html_parser = HTMLParser()
        self.control_char_regex = r'[\r\n\t]+'

    @property
    def is_retweet(self):
        return 'retweeted_status' in self.tweet

    @property
    def has_quoted_status(self):
        return 'quoted_status' in self.tweet

    def get_fields(self, fields):
        fields_obj = {}
        for f in fields:
            fields_obj[f] = self.get_field(f)
        return fields_obj

    def remove_control_characters(self, s):
        if not isinstance(s, str):
            return s
        # replace \t, \n and \r characters by a whitespace
        s = re.sub(self.control_char_regex, ' ', s)
        # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
        return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

    def get_field(self, field, fallback_to_non_str=True, silent_key_errors=True):
        """Get a first-level field"""
        if field == 'created_at':
            return pd.to_datetime(self.tweet['created_at'])
        elif field == 'text':
            return self.html_parser.unescape(self.get_text())
        elif field in ['id', 'in_reply_to_status_id', 'in_reply_to_user_id']:
            try:
                return self.tweet[field + '_str']
            except KeyError as e:
                if fallback_to_non_str:
                    try:
                        return str(self.tweet[field])
                    except KeyError as e:
                        if not silent_key_errors:
                            raise e
                else:
                    raise e
        else:
            field_val = self.tweet.get(field, None)
            if isinstance(field_val, str):
                field_val = self.remove_control_characters(field_val)
            return field_val

    def get_text(self):
        """Get full text (for both retweets and normal tweets)"""
        tweet_text = ''
        if self.is_retweet:
            prefix = self._get_retweet_prefix()
            tweet_text = prefix + self._get_full_text(self.tweet['retweeted_status'])
        else:
            tweet_text = self._get_full_text(self.tweet)
        return self.remove_control_characters(str(tweet_text))

    def get_coordinates_field(self):
        coordinates = {'has_coordinates': False}
        if 'coordinates' in self.tweet and self.tweet['coordinates'] is not None:
            coordinates['longitude'] = self.tweet['coordinates']['coordinates'][0]
            coordinates['latitude'] = self.tweet['coordinates']['coordinates'][1]
            coordinates['has_coordinates'] = True
        return coordinates

    def get_place_fields(self, keep_fields_place):
        def convert_to_polygon(s):
            if isinstance(s, list):
                for i, _s in enumerate(s):
                    s[i] = [float(_s[0]), float(_s[1])]
                return shapely.geometry.Polygon(s)
            else:
                return None
        place_obj = {'has_place': False, 'has_place_bounding_box': False,
                'place.bounding_box.centroid': None, 'place.bounding_box.area': None}
        if self.tweet['place'] is not None:
            place_obj['has_place'] = True
            for k in keep_fields_place:
                if k == 'bounding_box':
                    try:
                        place_obj['place.bounding_box'] = self.tweet['place'][k]['coordinates'][0]
                    except (KeyError, TypeError) as e:
                        pass
                    else:
                        place_obj['has_place_bounding_box'] = True
                        p = convert_to_polygon(place_obj['place.bounding_box'])
                        if isinstance(p, shapely.geometry.Polygon):
                            centroid = p.centroid
                            place_obj['place.bounding_box.centroid'] = [centroid.x, centroid.y]
                            place_obj['place.bounding_box.area'] = p.area
                else:
                    place_obj['place.' + k] = self.remove_control_characters(self.tweet['place'][k])
        return place_obj

    def get_user_fields(self, keep_fields_user):
        user_obj = {}
        for k in keep_fields_user:
            if k == 'id':
                user_obj['user.' + k] = self.remove_control_characters(self.tweet['user'].get(k + '_str', None))
            else:
                user_obj['user.' + k] = self.remove_control_characters(self.tweet['user'].get(k, None))
        return user_obj

    def get_entities_fields(self, keep_fields_entities):
        """Extract all entity information, take extended tweet if provided"""
        entities = {}
        for k in keep_fields_entities:
            if k == 'user_mentions' and 'user_mentions' in self.extended_tweet['entities']:
                entities['entities.' + k] = []
                for mention in self.extended_tweet['entities']['user_mentions']: 
                    entities['entities.' + k].append(mention['id_str'])
                if len(entities['entities.' + k]) == 0:
                    entities['entities.' + k] = None
            elif k == 'hashtags' and 'hashtags' in self.extended_tweet['entities']:
                entities['entities.' + k] = []
                for h in self.extended_tweet['entities']['hashtags']:
                    entities['entities.' + k].append(self.remove_control_characters(h['text']))
                if len(entities['entities.' + k]) == 0:
                    entities['entities.' + k] = None
            else:
                entities['entities.' + k] = self.extended_tweet['entities'].get(k, None)
        return entities

    def get_retweet_info(self, keep_fields_retweeted_status):
        retweet_info = {'is_retweet': self.is_retweet, **{k: None for k in keep_fields_retweeted_status}}
        if self.is_retweet:
            for field in keep_fields_retweeted_status:
                if field == 'retweeted_status.favorite_count':
                    if 'favorite_count' in self.tweet['retweeted_status']:
                        retweet_info[field] = self._extract_subfield(field)
                else:
                    retweet_info[field] = self._extract_subfield(field)
        return retweet_info

    def get_quoted_status_info(self, keep_fields_quoted_status):
        quoted_status_info = {
                'has_quoted_status': self.has_quoted_status, 
                'quoted_status.has_media': False,
                'quoted_status.media': {},
                **{f: None for f in keep_fields_quoted_status}
                }
        if not self.has_quoted_status:
            return quoted_status_info
        else:
            for field in keep_fields_quoted_status:
                if field == 'quoted_status.text':
                    quoted_status_info[field] = self.remove_control_characters(self._get_full_text(self.tweet['quoted_status']))
                else:
                    quoted_status_info[field] = self._extract_subfield(field)
            media_info = self.get_media_info(tweet_obj=self.tweet['quoted_status'])
            quoted_status_info['quoted_status.has_media'] = media_info['has_media']
            quoted_status_info['quoted_status.media'] = media_info['media']
        return quoted_status_info

    def get_media_info(self, tweet_obj=None):
        if tweet_obj is None:
            tweet_obj = self.tweet
        media_info = {'has_media': False, 'media': {}, 'media_image_urls': []}
        if self._keys_exist(tweet_obj, 'extended_tweet', 'extended_entities', 'media'):
            tweet_media = tweet_obj['extended_tweet']['extended_entities']['media']
        elif self._keys_exist(tweet_obj, 'extended_entities', 'media'):
            tweet_media = tweet_obj['extended_entities']['media']
        else:
            return media_info
        media_info['has_media'] = True
        media_info['media'] = defaultdict(lambda: 0)
        for m in tweet_media:
            media_info['media'][m['type']] += 1
            # for media of type video/animated_gif media_url corresponds to a thumbnail image
            media_info['media_image_urls'].append(m['media_url'])
        media_info['media'] = dict(media_info['media'])
        return media_info

    def replace_user_mentions(self, tweet_text, status_type='default'):
        """Replaces @user mentions in tweet text based on indices provided in entities.user_mentions.indices"""
        filler = '@<user>'
        corr = 0
        if status_type == 'default':
            try:
                user_mentions = self.extended_tweet['entities']['user_mentions'] 
            except KeyError:
                user_mentions = []
        if status_type == 'quoted':
            if 'extended_tweet' in self.tweet['quoted_status']:
                if 'extended_tweet' in self.tweet['quoted_status']:
                    user_mentions = self.tweet['quoted_status']['extended_tweet']['entities']['user_mentions'] 
                else:
                    user_mentions = self.tweet['quoted_status']['entities']['user_mentions'] 
            else:
                user_mentions = self.tweet['quoted_status']['entities']['user_mentions'] 
        for m in user_mentions:
            s, e = m['indices']
            s -= corr
            e -= corr
            tweet_text = tweet_text[:s] + filler + tweet_text[e:] 
            corr += (e-s) - len(filler)
        return tweet_text

    def replace_urls(self, tweet_text):
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>', tweet_text)

    def anonymize_text(self, tweet_text, status_type='default'):
        tweet_text = self.replace_user_mentions(tweet_text, status_type)
        tweet_text = self.replace_urls(tweet_text)
        return tweet_text

    def anonymize(self, tweet_obj):
        tweet_obj_anonymized = copy(tweet_obj)
        tweet_obj_anonymized['text'] = self.anonymize_text(tweet_obj_anonymized['text'])
        if tweet_obj_anonymized['has_quoted_status']:
            tweet_obj_anonymized['quoted_status.text'] = self.anonymize_text(tweet_obj_anonymized['quoted_status.text'], status_type='quoted')
        return tweet_obj_anonymized
        
    def contains_keywords(self, keywords, search_text=True, search_urls=True):
        """Here we pool all relevant text within the tweet to do the matching. From the twitter docs:
        "Specifically, the text attribute of the Tweet, expanded_url and display_url for links and media, text for hashtags, and screen_name for user mentions are checked for matches."
        https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/basic-stream-parameters.html
        """
        # anything which is inside text (user_mentions, hashtags etc.) will be tested for a raw match
        any_match_text = ''
        if search_text:
            any_match_text += self.get_text()
            any_match_text += self._fetch_user_mentions()
            any_match_text = any_match_text.lower()
        # for URLS only match if surrounded by non-alphabetic characters
        separator_match_text = ''
        if search_urls:
            separator_match_text += self._fetch_urls()
            separator_match_text = separator_match_text.lower()
        for keyword in keywords:
            m = re.search(r'{}'.format(keyword), any_match_text)
            if m is not None:
                return True
            m = re.search(r'(\b|\d|_){}(\b|\d|_|cas)'.format(keyword), separator_match_text)
            if m is not None:
                return True
        return False

    # private methods

    def _fetch_urls(self):
        t = []
        if self.is_retweet:
            tweet_obj = self.tweet['retweeted_status']
        else:
            tweet_obj = self.tweet
        if 'urls' in tweet_obj['entities']:
            for u in tweet_obj['entities']['urls']:
                if 'unwound' in u:
                    if u['unwound']['url'] is not None:
                        t.append(u['unwound']['url'])
                if u['expanded_url'] is not None:
                    t.append(u['expanded_url'])
        if 'extended_entities' in tweet_obj:
            if 'media' in tweet_obj['extended_entities']:
                for m in tweet_obj['extended_entities']['media']:
                    if m['expanded_url'] is not None:
                        t.append(m['expanded_url'])
        return ' '.join(t)

    def _fetch_user_mentions(self):
        t = []
        if self.is_retweet:
            tweet_obj = self.tweet['retweeted_status']
        else:
            tweet_obj = self.tweet
        if 'user_mentions' in tweet_obj['entities']:
            for user_mention in tweet_obj['entities']['user_mentions']:
                t.append(user_mention['screen_name'])
        return ' '.join(t)

    def _extract_subfield(self, field):
        subf = field.split('.')
        if subf[-1] in ['id', 'in_reply_to_status_id']:
            subf[-1] += '_str'
        if len(subf) == 2:
            return self.tweet[subf[0]][subf[1]]
        elif len(subf) == 3:
            return self.tweet[subf[0]][subf[1]][subf[2]]
        else:
            raise Exception('Number of subfields is too deep')

    def _get_full_text(self, tweet_obj):
        if 'extended_tweet' in tweet_obj:
            return tweet_obj['extended_tweet']['full_text']
        else:
            return tweet_obj['text']

    def _get_retweet_prefix(self):
        m = re.match(r'^RT (@\w+): ', self.tweet['text'])
        try:
            return m[0]
        except TypeError:
            return ''

    def _get_extended_tweet(self):
        if 'extended_tweet' in self.tweet:
            return self.tweet['extended_tweet']
        else:
            return self.tweet

    def _keys_exist(self, element, *keys):
        """ Check if *keys (nested) exists in `element` (dict). """
        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True


def process_file(f_name, config):
    all_data = {output_type: [] for output_type in config.output_types}
    count = 0
    if config.output_types.encrypted:
        encrypt = Encrypt()
    with open(f_name, 'r') as f:
        num_lines = sum(1 for line in f)
        f.seek(0)
        for line in f:
            if len(line) <= 1:
                continue
            try:
                tweet = json.loads(line)
            except:
                # some files use single quotation, for this we need to use ast.literal_eval
                tweet = ast.literal_eval(line)
            if 'info' in tweet:
                continue  # API log fields (can be ignored)
            pt = MergeTweet(tweet)
            tweet_obj = {
                    **pt.get_fields(config.keep_fields),
                    **pt.get_user_fields(config.keep_fields_user),
                    **pt.get_entities_fields(config.keep_fields_entities),
                    **pt.get_place_fields(config.keep_fields_place),
                    **pt.get_coordinates_field(),
                    **pt.get_retweet_info(config.keep_fields_retweeted_status),
                    **pt.get_quoted_status_info(config.keep_fields_quoted_status),
                    **pt.get_media_info(),
                    'extracted_quoted_tweet': False,
                    'contains_keywords': pt.contains_keywords(config.keywords)
                    }
            if config.output_types.original:
                all_data['original'].append(tweet_obj)
            if config.output_types.anonymized:
                tweet_obj_anonymized = pt.anonymize(tweet_obj)
                all_data['anonymized'].append(tweet_obj_anonymized)
            if config.output_types.encrypted:
                if not config.output_types.anonymized:
                    tweet_obj_anonymized = pt.anonymize(tweet_obj)
                all_data['encrypted'].append(encrypt.encode_tweet(copy(tweet_obj_anonymized), config.encrypt_fields))
            count +=1
            # append quoted tweet as normal tweet
            if tweet_obj['has_quoted_status']:
                pt = MergeTweet(tweet['quoted_status'])
                tweet_obj = {
                        **pt.get_fields(config.keep_fields),
                        **pt.get_user_fields(config.keep_fields_user),
                        **pt.get_entities_fields(config.keep_fields_entities),
                        **pt.get_place_fields(config.keep_fields_place),
                        **pt.get_coordinates_field(),
                        **pt.get_retweet_info(config.keep_fields_retweeted_status),
                        **pt.get_quoted_status_info(config.keep_fields_quoted_status),
                        **pt.get_media_info(),
                        'extracted_quoted_tweet': True,
                        'contains_keywords': pt.contains_keywords(config.keywords)
                        }
                if config.output_types.original:
                    all_data['original'].append(tweet_obj)
                if config.output_types.anonymized:
                    tweet_obj_anonymized = pt.anonymize(tweet_obj)
                    all_data['anonymized'].append(tweet_obj_anonymized)
                if config.output_types.encrypted:
                    if not config.output_types.anonymized:
                        tweet_obj_anonymized = pt.anonymize(tweet_obj)
                    all_data['encrypted'].append(encrypt.encode_tweet(copy(tweet_obj_anonymized), config.encrypt_fields))
                count +=1
    return all_data
    

def run(dtypes=['original'], yearly=True, no_parallel=False, verbose=False):
    logger = logging.getLogger(__name__)
    # build config
    config = DefaultMunch(None)
    config.input_data_path = os.path.join('data', '0_raw')
    config.output_data_path = os.path.join('data', '1_merged')
    # fields to keep
    config.keep_fields = ['id', 'created_at', 'text', 'in_reply_to_status_id', 'in_reply_to_user_id', 'reply_count', 'retweet_count', 'favorite_count']
    config.keep_fields_user = ['id', 'screen_name', 'name', 'location', 'followers_count', 'friends_count']
    config.keep_fields_entities = ['hashtags', 'user_mentions']
    config.keep_fields_place = ['bounding_box', 'full_name', 'country_code', 'place_type']
    config.keep_fields_quoted_status = ['quoted_status.id', 'quoted_status.text', 'quoted_status.user.id', 'quoted_status.user.followers_count','quoted_status.in_reply_to_status_id', 'quoted_status.retweet_count', 'quoted_status.favorite_count']
    config.keep_fields_retweeted_status = ['retweeted_status.id', 'retweeted_status.user.id', 'retweeted_status.user.followers_count', 'retweeted_status.in_reply_to_status_id', 'retweeted_status.retweet_count', 'retweeted_status.favorite_count']
    config.encrypt_fields = DefaultMunch.fromDict({'entities.user_mentions': list, 'id': str, 'in_reply_to_status_id': str, 
                      'in_reply_to_user_id': str, 'user.id': str, 'user.screen_name': str, 'user.name': str,
                      'entities.user_mentions': list, 'retweeted_status.id': str, 'retweeted_status.user.id': str, 'retweeted_status.in_reply_to_status_id': str,
                      'quoted_status.id': str, 'quoted_status.user.id': str, 'quoted_status.in_reply_to_status_id': str}, None)
    project_info = utils.helpers.get_project_info()
    config.keywords = project_info['keywords']
    # params
    config.output_types = DefaultMunch.fromDict({dtype: True for dtype in dtypes}, False)
    config.output_intervals = DefaultMunch.fromDict({'all': True, 'yearly': yearly}, False)
    # make sure encryption key is set
    if config.output_types.encrypted:
        Encrypt.verify_encryption_key()
    # run
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    s_time = time.time()
    parallel = joblib.Parallel(n_jobs=num_cores)
    process_file_delayed = joblib.delayed(process_file)
    data_files = sorted([f for f in glob.glob(os.path.join(config.input_data_path, '**', '*.json*'), recursive=True) if os.path.isfile(f)])
    all_data = parallel((process_file_delayed(f_name, config) for f_name in tqdm(data_files)))
    # merge
    all_data = {dtype: list(itertools.chain.from_iterable([d[dtype] for d in all_data])) for dtype in config.output_types}
    data_size = set()
    # write output files
    for t in config.output_types:
        if config.output_types[t]:
            df = pd.DataFrame(all_data[t])
            # Drop dulicates by ID
            df.drop_duplicates(subset='id', keep='first', inplace=True)
            logger.info('Writing data of type {} ...'.format(t))
            df['created_at'] = pd.to_datetime(df['created_at'])
            df.sort_values('created_at', inplace=True)
            df.set_index('created_at', inplace=True, drop=False)
            data_size.add(len(df))
            if config.output_intervals.all:
                df.to_csv(os.path.join(config.output_data_path, 'merged_{}.csv'.format(t)), index=False, encoding='utf8') 
            if config.output_intervals.yearly:
                date_ranges = [['{}-01-01'.format(y), '{}-12-31'.format(y), y] for y in range(2000, 2040)]
                for date_range in date_ranges:
                    if len(df[date_range[0]:date_range[1]]) > 0:
                        df[date_range[0]:date_range[1]].to_csv(os.path.join(config.output_data_path, 'merged_{}_year_{}.csv'.format(t, date_range[2])), index=False, encoding='utf8')   
    e_time = time.time()
    if len(data_size) > 1:
        warnings.warn('Collected datatypes are of different sizes. This should normally not happen.')
    data_size = data_size.pop()
    logger.info('Merged {:,} tweets after {:.1f} min'.format(data_size, (e_time - s_time)/60.0))
