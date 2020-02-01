import pandas as pd
import os
import pickle
import logging
from preprocess.utils.helpers import get_parsed_data, get_cache_path, get_data_folder
import numpy as np
from tqdm import tqdm
import numpy as np
import multiprocessing
import joblib
import ast
import shapely.geometry

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s') 
log = logging.getLogger(__name__)

class GeoEnrichment():
    """
    GeoEnrichment

    Download geo dataset from: 
    - https://download.geonames.org/export/dump/allCountries.zip
    - https://download.geonames.org/export/dump/featureCodes_en.txt
    """

    def __init__(self):
        self.columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1', 'admin2', 'admin3', 'admin4', 'population', 'elevation', 'dem', 'timezone', 'modification_date']
        self.raw_data = os.path.join(get_data_folder(), 'geodata', 'allCountries.txt')
        self.feature_code_path = os.path.join(get_data_folder(), 'geodata', 'featureCodes_en.txt')
        self.parsed_data_cols = ['id', 'has_place_bounding_box', 'has_coordinates', 'user.location', 'place.bounding_box.centroid', 'longitude', 'latitude']

    def get_geo_enriched_data(self, dtype='anonymized', with_altnames=False, use_cache=True, use_cache_geo_data=True, use_cache_keyword_processor=True, no_parallel=True, nrows_input_data=None):
        """
        Infers 3 types of geo enrichement:
        1) 'coordinates': Coordinates were provided in tweet object
        2) 'place_centroid': Compute centroid of place field
        3) 'user_location': Infer geo-location from user location

        Method returns pandas DataFrame with 3 new columns which the same number of rows as provided from `get_parsed_data` but with 3 new columns:
        - `longitude_enriched`
        - `latitude_enriched`
        - `geo_enriched_type`
        """
        with_altnames_tag = '_with_altnames' if with_altnames else ''
        nrows_tag = '' if nrows_input_data is None else str(nrows_input_data)
        cache_path = get_cache_path(f'geonames_enriched_{dtype}{with_altnames_tag}{nrows_tag}.pkl')
        if not use_cache or not os.path.isfile(cache_path):
            log.info(f'Computing geo-enriched data for type {dtype}...')
            log.info(f'Load data...')
            df = get_parsed_data(dtype=dtype, usecols=self.parsed_data_cols, nrows=nrows_input_data)
            geo_data = self.get_geo_data(use_cache=use_cache_geo_data, with_altnames=with_altnames)
            kp = self.get_keyword_processor(use_cache=use_cache_keyword_processor, with_altnames=with_altnames, geo_data=geo_data)
            # enrich geo-information
            df['longitude_enriched'] = np.nan
            df['latitude_enriched'] = np.nan
            df['geo_enriched_type'] = ''
            # fill geo coordinates
            df.loc[df.has_coordinates, 'longitude_enriched'] = df.loc[df.has_coordinates, 'longitude']
            df.loc[df.has_coordinates, 'latitude_enriched'] = df.loc[df.has_coordinates, 'latitude']
            df.loc[df.has_coordinates, 'geo_enriched_type'] = 'coordinates'
            # fill place centroid coordinates
            df['place.bounding_box.centroid'] = df['place.bounding_box.centroid'].apply(self.convert_to_coordinate_from_list)
            row_selection = (~df.has_coordinates) & (df.has_place_bounding_box)
            df.loc[row_selection, 'longitude_enriched'] = df.loc[row_selection, 'place.bounding_box.centroid'].apply(lambda s: s.x)
            df.loc[row_selection, 'latitude_enriched'] = df.loc[row_selection, 'place.bounding_box.centroid'].apply(lambda s: s.y)
            df.loc[row_selection, 'geo_enriched_type'] = 'place_centroid'
            # fill from user location in parallel
            if no_parallel:
                num_cores = 1
            else:
                num_cores = max(multiprocessing.cpu_count() - 1, 1)
            parallel = joblib.Parallel(n_jobs=num_cores)
            reverse_match_user_location_delayed = joblib.delayed(self.reverse_match_user_location)
            row_selection = (~df.has_coordinates) & (~df.has_place_bounding_box) & (~df['user.location'].isnull())
            _df = df.loc[row_selection, ['user.location', 'longitude_enriched', 'latitude_enriched', 'geo_enriched_type']]
            num_splits = min(max(len(_df) // 1000, 1), len(_df))
            log.info(f'Using {num_cores} cores to run {num_splits} jobs...')
            _df = np.array_split(_df, num_splits)
            res = parallel((reverse_match_user_location_delayed(batch, kp, geo_data) for batch in tqdm(_df)))
            log.info('Merging...')
            res = pd.concat(res, axis=0)
            col_selection = ['longitude_enriched', 'latitude_enriched', 'geo_enriched_type']
            df.loc[row_selection, col_selection] = res.loc[:, col_selection]
            log.info(f'Writing to pickle file {cache_path}...')
            df[col_selection].to_pickle(cache_path)
        log.info(f'Reading geo enriched data of type {dtype}...')
        df = pd.read_pickle(cache_path)
        return df

    def reverse_match_user_location(self, df, kp, geo_data):
        for i, row in df.iterrows():
            try:
                # find corresponding coordinates
                matches = self.match_location(row['user.location'], kp, geo_data)
            except Exception as e:
                print(e)
            else:
                if len(matches) == 0:
                    continue
                df.loc[i, 'longitude_enriched'] = matches[0][1]
                df.loc[i, 'latitude_enriched'] = matches[0][2]
                df.loc[i, 'geo_enriched_type'] = 'user_location'
        return df

    def match_location(self, location, kp, geo_data):
        # find matches and return list of places
        matches = kp.extract_keywords(location)
        if len(matches) == 0:
            return []
        # sort by priorities
        matches = sorted([int(m) for m in matches])
        return [geo_data[m] for m in matches]

    def get_geo_data(self, use_cache=True, with_altnames=False):
        """
        Transform raw data of country names into a list of coordinates (where the position i corresponds to the priority i of the element)
        """
        with_altnames_tag = '_with_altnames' if with_altnames else ''
        cache_path = get_cache_path(f'geonames{with_altnames_tag}.pkl')
        if not os.path.isfile(cache_path) or not use_cache:
            log.info('Transforming geonames data...')
            log.info('Reading geo data...')
            dtypes = {'name': str, 'latitude': float, 'longitude': float, 'country_code': str, 'population': int, 'feature_code': str}
            if with_altnames:
                dtypes['alternatenames'] = str
            df = pd.read_csv(self.raw_data, names=self.columns, sep='\t', dtype=dtypes, usecols=dtypes.keys())
            # select places with a population greater zero
            df = df[df.population > 0]
            # get rid of administrative zones without country codes (e.g. "The Commonwealth")
            df = df[~df.country_code.isnull()]
            # select places with feature class A (admin) and P (place)
            df_features = pd.read_csv(self.feature_code_path, sep='\t', names=['feature_code', 'description-short', 'description-long'])
            df_features['feature_code_class'] = ''
            df_features.loc[:, ['feature_code_class', 'feature_code']] = df_features.feature_code.str.split('.', expand=True).values
            df = df.merge(df_features, on='feature_code', how='left')
            df = df[df.feature_code_class.isin(['A', 'P'])]
            # generate list of priorities by ID. Levels of priority:
            # - Places are given priority over admin areas
            # - Among places, sort by "importance of a place" (PPL > PPLA > PPLA2, etc.)
            # - Among admin give priority to second order divisions over country names, after that priority decreases with area size (ADM2 > ADM1 > ADM3 > ADM4), 
            # - Among feature class prioritize by population size
            log.info('Sorting by priority...')
            feature_code_priorities = ['PPL', 'PPLA', 'PPLA2', 'PPLA3', 'PPLA4', 'PPLA5', 'PPLC', 'PPLCH', 'PPLF', 'PPLG', 'PPLH', 'PPLL',
                    'PPLQ', 'PPLR', 'PPLS', 'PPLW', 'PPLX', 'STLMT', 'ADM2', 'ADM2H', 'ADM1', 'ADM1H', 'ADM3', 'ADM3H', 'ADM4', 'ADM4H',
                    'ADM5', 'ADM5H', 'ADMD', 'ADMDH', 'LTER', 'PCL', 'PCLD', 'PCLF', 'PCLH', 'PCLI', 'PCLIX', 'PCLS', 'PRSH', 'TERR', 'ZN', 'ZNB']
            feature_code_priorities = {k: i for i, k in enumerate(feature_code_priorities)}
            df['priority'] = df.feature_code.apply(lambda code: feature_code_priorities[code])
            df.sort_values(by=['priority', 'population'], ascending=[True, False], inplace=True)
            geo_coords = []
            # Build geo data array. Position i in list corresponds to priority
            log.info('Build coordinate data list...')
            for _, row in tqdm(df.iterrows(), total=len(df)):
                geo_coords.append([row['name'], row['longitude'], row['latitude']])
                if with_altnames:
                    if isinstance(row['alternatenames'], str):
                        altnames = row['alternatenames'].split(',')
                        for altname in altnames:
                            geo_coords.append([altname, row['longitude'], row['latitude']])
            log.info('Writing to pickle...')
            with open(cache_path, 'wb') as f:
                pickle.dump(geo_coords, f)
            log.info('...done')
        log.info('Reading cached geo data...')
        with open(cache_path, 'rb') as f:
           geo_coords = pickle.load(f)
        return geo_coords

    def get_keyword_processor(self, use_cache=True, with_altnames=False, geo_data=None):
        """Use flashtext to build trie of names of locations which map to corresponding ID"""
        with_altnames_tag = '_with_altnames' if with_altnames else ''
        cache_path = get_cache_path(f'geonames_keyword_processor{with_altnames_tag}.pkl')
        if not os.path.isfile(cache_path) or not use_cache:
            from flashtext import KeywordProcessor
            if geo_data is None:
                geo_data = self.get_geo_data(with_altnames=with_altnames)
            kp = KeywordProcessor()
            log.info('Adding terms to keyword processor (building trie)...')
            for i, item in tqdm(enumerate(geo_data), total=len(geo_data)):
                idx = str(i)
                kp.add_keyword(item[0], idx)
            with open(cache_path, 'wb') as f:
                pickle.dump(kp, f)
        log.info('Reading cached keyword processor...')
        with open(cache_path, 'rb') as f:
           kp = pickle.load(f)
        return kp

    def convert_to_coordinate_from_list(self, l):
        """Takes stringified python list of longitude/latitude pair and returns shapely Point"""
        if pd.isna(l):
            return np.nan
        else:
            l = ast.literal_eval(l)
            return shapely.geometry.Point(l[0], l[1])

if __name__ == "__main__":
    ge = GeoEnrichment()
    df = ge.get_geo_enriched_data(use_cache=True, with_altnames=False, use_cache_keyword_processor=True, use_cache_geo_data=True, nrows_input_data=None)
    __import__('pdb').set_trace()
