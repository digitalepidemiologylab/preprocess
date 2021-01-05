import pandas as pd
from copy import copy
from collections import defaultdict
import spacy
import hashlib
import shapely.geometry
import pickle
import itertools
import unicodedata
import re
import logging
from functools import lru_cache
import html

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')

# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
control_char_regex = re.compile(r'[\r\n\t]+')

class ProcessTweet():
    """Wrapper class for functions to process/modify tweets"""

    def __init__(self, tweet=None, keywords=None, map_data=None, gc=None):
        self.tweet = tweet
        self.extended_tweet = self._get_extended_tweet()
        if keywords is None:
            self.keywords = []
        else:
            self.keywords = keywords
        self.map_data = map_data
        self.gc = gc

    @property
    def id(self):
        return self.tweet['id_str']

    @property
    def retweeted_status_id(self):
        return self.tweet['retweeted_status']['id_str']

    @property
    def retweeted_user_id(self):
        return self.tweet['retweeted_status']['user']['id_str']

    @property
    def quoted_status_id(self):
        return self.tweet['quoted_status']['id_str']

    @property
    def quoted_user_id(self):
        return self.tweet['quoted_status']['user']['id_str']

    @property
    def replied_status_id(self):
        return self.tweet['in_reply_to_status_id_str']

    @property
    def replied_user_id(self):
        return self.tweet['in_reply_to_user_id_str']

    @property
    def is_retweet(self):
        return 'retweeted_status' in self.tweet

    @property
    def has_quote(self):
        return 'quoted_status' in self.tweet

    @property
    def is_reply(self):
        return self.tweet['in_reply_to_status_id_str'] is not None

    @property
    def user_id(self):
        return self.tweet['user']['id_str']

    @property
    def user_timezone(self):
        try:
            return self.tweet['user']['timezone']
        except KeyError:
            return None

    @property
    def is_verified(self):
        return self.tweet['user']['verified']

    @property
    def has_coordinates(self):
        return 'coordinates' in self.tweet and self.tweet['coordinates'] is not None

    @property
    def has_place(self):
        return self.tweet['place'] is not None

    @property
    def lang(self):
        try:
            return self.tweet['lang']
        except KeyError:
            return None

    @property
    def has_place_bounding_box(self):
        if not self.has_place:
            return False
        try:
            self.tweet['place']['bounding_box']['coordinates'][0]
        except (KeyError, TypeError):
            return False
        else:
            return True

    @lru_cache(maxsize=1)
    def get_text(self):
        """Get full text"""
        tweet_obj = self.tweet
        if self.is_retweet:
            # in retweets text is usually truncated, therefore get the text from original status
            tweet_obj = self.tweet['retweeted_status']
        if 'extended_tweet' in tweet_obj:
            text = tweet_obj['extended_tweet']['full_text']
        else:
            text = tweet_obj['text']
        return ProcessTweet.normalize_str(text)

    @staticmethod
    def replace_usernames(text, filler='@user'):
        # replace other user handles by filler
        text = re.sub(username_regex, filler, text)
        # add spaces between, and remove double spaces again
        text = text.replace(filler, f' {filler} ')
        text = ' '.join(text.split())
        return text

    @staticmethod
    def anonymize_text(text, url_filler='<url>', user_filler='@user'):
        text = ProcessTweet.replace_urls(text, filler=url_filler)
        text = ProcessTweet.replace_usernames(text, filler=user_filler)
        return text

    @staticmethod
    def normalize_str(s):
        if not s:
            return ''
        if not isinstance(s, str):
            s = str(s)
        # covnert HTML
        s = html.unescape(s)
        # replace \t, \n and \r characters by a whitespace
        s = re.sub(control_char_regex, ' ', s)
        # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
        s =  "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C')
        # remove duplicate whitespace
        s = ' '.join(s.split())
        return s

    @staticmethod
    def replace_urls(text, filler='<url>'):
        # replace other urls by filler
        text = re.sub(url_regex, filler, text)
        # add spaces between, and remove double spaces again
        text = text.replace(filler, f' {filler} ')
        text = ' '.join(text.split())
        return text

    def convert_to_iso_time(self, date):
        ts = pd.to_datetime(date)
        return ts.isoformat()

    def extract(self, tweet_type='original'):
        geo_obj = self.get_geo_info()
        return {
                'id': self.id,
                'text': self.get_text(),
                'in_reply_to_status_id': self.replied_status_id,
                'in_reply_to_user_id': self.replied_user_id,
                'quoted_user_id': self.quoted_user_id if self.has_quote else None,
                'quoted_status_id': self.quoted_status_id if self.has_quote else None,
                'retweeted_user_id': self.retweeted_user_id if self.is_retweet else None,
                'retweeted_status_id': self.retweeted_status_id if self.is_retweet else None,
                'created_at': self.convert_to_iso_time(self.tweet['created_at']),
                'entities.user_mentions': self.get_user_mentions(),
                'user.id': self.user_id,
                'user.screen_name': self.tweet['user']['screen_name'],
                'user.name': self.tweet['user']['name'],
                'user.description': ProcessTweet.normalize_str(self.tweet['user']['description']),
                'user.timezone': self.user_timezone,
                'user.location': self.tweet['user']['location'],
                'user.num_followers': self.tweet['user']['followers_count'],
                'user.num_following': self.tweet['user']['friends_count'],
                'user.created_at': self.convert_to_iso_time(self.tweet['user']['created_at']),
                'user.statuses_count': self.tweet['user']['statuses_count'],
                'user.is_verified': self.is_verified,
                'lang': self.lang,
                'token_count': self.get_token_count(),
                'is_retweet': self.is_retweet,
                'has_quote': self.has_quote,
                'is_reply': self.is_reply,
                'contains_keywords': self.contains_keywords(),
                **geo_obj
                }

    def get_geo_info(self):
        """
        Tries to infer different types of geoenrichment from tweet (ProcessTweet object)
        Returns dictionary with the following keys:
        - longitude (float)
        - latitude (float)
        - country_code (str)
        - geoname_id (str): unique identifier given for a location on geonames.org
        - location_type (str): either refers to place_type provided by the Twitter "Place" object, or location_type provided by the local-geocode package
        - geo_type (int): specifies four ways of geolocation extraction:
            0: no geolocation could be inferred
            1: exact geocoordinates provided by Twitter
            2: Place Polygon provided by Twitter [(longitude, latitude) refers to the centroid of the bounding box]
            3: geolocation parsed from user.location field with local-geocode [in this case, geoname_id is also provided]
        - region (str)
        - subregion (str)
        Regions (according to World Bank):
        East Asia & Pacific, Latin America & Caribbean, Europe & Central Asia, South Asia,
        Middle East & North Africa, Sub-Saharan Africa, North America, Antarctica
        Subregions:
        South-Eastern Asia, South America, Western Asia, Southern Asia, Eastern Asia, Eastern Africa,
        Northern Africa Central America, Middle Africa, Eastern Europe, Southern Africa, Caribbean,
        Central Asia, Northern Europe, Western Europe, Southern Europe, Western Africa, Northern America,
        Melanesia, Antarctica, Australia and New Zealand, Polynesia, Seven seas (open ocean), Micronesia
        """
        def get_region_by_country_code(country_code):
            return self.map_data[self.map_data['ISO_A2'] == country_code].iloc[0].REGION_WB

        def get_subregion_by_country_code(country_code):
            return self.map_data[self.map_data['ISO_A2'] == country_code].iloc[0].SUBREGION

        def get_country_code_by_coords(longitude, latitude):
            coordinates = shapely.geometry.point.Point(longitude, latitude)
            within = self.map_data.geometry.apply(lambda p: coordinates.within(p))
            if sum(within) > 0:
                return self.map_data[within].iloc[0].ISO_A2
            else:
                dist = self.map_data.geometry.apply(lambda poly: poly.distance(coordinates))
                closest_country = self.map_data.iloc[dist.argmin()].ISO_A2
                logger.warning(f'Coordinates {longitude}, {latitude} were outside of a country land area but were matched to closest country ({closest_country})')
                return closest_country

        def convert_to_polygon(s):
            for i, _s in enumerate(s):
                s[i] = [float(_s[0]), float(_s[1])]
            return shapely.geometry.Polygon(s)

        geo_obj = {
                'longitude': None,
                'latitude': None,
                'country_code': None,
                'geoname_id': None,
                'location_type': None,
                'geo_type': 0,
                'region': None,
                'subregion': None
                }
        if self.has_coordinates:
            # try to get geo data from coordinates (<0.1% of tweets)
            geo_obj['longitude'] = float(self.tweet['coordinates']['coordinates'][0])
            geo_obj['latitude'] = float(self.tweet['coordinates']['coordinates'][1])
            geo_obj['country_code'] = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
            geo_obj['geo_type'] = 1
        elif self.has_place_bounding_box:
            # try to get geo data from place (roughly 1% of tweets)
            p = convert_to_polygon(self.tweet['place']['bounding_box']['coordinates'][0])
            geo_obj['longitude'] = p.centroid.x
            geo_obj['latitude'] = p.centroid.y
            country_code = self.tweet['place']['country_code']
            if country_code == '':
                # sometimes places don't contain country codes, try to resolve from coordinates
                country_code = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
            geo_obj['country_code'] = country_code
            geo_obj['location_type'] = self.tweet['place']['place_type']
            geo_obj['geo_type'] = 2
        
        else:
            # try to parse user location
            locations = self.gc.decode(self.tweet['user']['location'])
            if len(locations) > 0:
                geo_obj['longitude'] = locations[0]['longitude']
                geo_obj['latitude'] = locations[0]['latitude']
                country_code = locations[0]['country_code']
                if country_code == '':
                    # sometimes country code is missing (e.g. disputed areas), try to resolve from geodata
                    country_code = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
                geo_obj['country_code'] = country_code
                geo_obj['geoname_id'] = locations[0]['geoname_id']
                geo_obj['location_type'] = locations[0]['location_type']
                geo_obj['geo_type'] = 3

        if not pd.isna(geo_obj['country_code']):
            # retrieve region info
            if geo_obj['country_code'] in self.map_data.ISO_A2.tolist():
                geo_obj['region'] = get_region_by_country_code(geo_obj['country_code'])
                geo_obj['subregion'] = get_subregion_by_country_code(geo_obj['country_code'])
            else:
                logger.warning(f'Unknown country_code {geo_obj["country_code"]}')
        return geo_obj

    def get_user_mentions(self):
        user_mentions = []
        if 'user_mentions' in self.extended_tweet['entities']:
            for mention in self.extended_tweet['entities']['user_mentions']:
                user_mentions.append(mention['id_str'])
        if len(user_mentions) == 0:
            return None
        else:
            return user_mentions

    @property
    def has_media(self):
        try:
            self.extended_tweet['extended_entities']['media']
        except KeyError:
            return False
        else:
            return True

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
        """Replaces @user mentions in tweet text based on indices provided in entities.user_mentions.indices.
        This method is arguably more complext than a simple regex but it relies on actually tagged users.
        """
        filler = '@user'
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

    def contains_keywords(self, search_text=True, search_urls=True):
        """Here we pool all relevant text within the tweet to do the matching. From the twitter docs:
        "Specifically, the text attribute of the Tweet, expanded_url and display_url for links and media, text for hashtags, and screen_name for user mentions are checked for matches."
        https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/basic-stream-parameters.html
        """
        if len(self.keywords) == 0:
            return False
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
        for keyword in self.keywords:
            m = re.search(r'{}'.format(keyword), any_match_text)
            if m is not None:
                return True
            m = re.search(r'(\b|\d|_){}(\b|\d|_|cas)'.format(keyword), separator_match_text)
            if m is not None:
                return True
        return False

    def get_token_count(self):
        text = self.get_text()
        # remove user handles and URLs from text
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
        text = re.sub('(\@[^\s]+)', '', text)
        text = text.strip()
        doc = nlp(text, disable=['parser', 'tagger', 'ner'])
        # Count the number of tokens excluding stopwords
        token_count = len([token for token in doc if token.is_alpha and not token.is_stop])
        return token_count

    def get_text_hash(self):
        return hashlib.md5(self.get_text().encode('utf-8')).hexdigest()

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

