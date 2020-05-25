import os
import geopandas as gpd
import urllib.request
import zipfile
import logging
import shapely
from utils.helpers import find_project_root

logger = logging.getLogger(__name__)


def load_map_data():
    geo_data_folder = os.path.join(find_project_root(),  'data', 'other', 'geodata')
    map_data_path = os.path.join(geo_data_folder, 'ne_10m_admin_0_countries.shp')
    if not os.path.isfile(map_data_path):
        # download & unzip data
        if not os.path.isdir(geo_data_folder):
            os.makedirs(geo_data_folder)
        logger.info(f'Downloading country shapefile to {geo_data_folder}')
        zipfile_path = os.path.join(geo_data_folder, 'geodata.zip')
        urllib.request.urlretrieve(
                'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip',
                zipfile_path)
        logger.info('Extracting...')
        with zipfile.ZipFile(zipfile_path, 'r') as f:
            f.extractall(geo_data_folder)
        os.remove(zipfile_path)
    df = gpd.read_file(map_data_path, encoding='utf-8')
    df.crs = 'epsg:4326'
    # only keep certain columns
    df = df[['ISO_A2','ISO_A3', 'NAME_EN', 'REGION_WB', 'SUBREGION', 'geometry']]
    # ISO A2 codes are not set for France & Norway for some reason...
    df.loc[df.NAME_EN == 'France', 'ISO_A2'] = 'FR'
    df.loc[df.NAME_EN == 'Norway', 'ISO_A2'] = 'NO'
    # Add CS (old code for Serbia & Montenegro, still used by Twitter)
    serbia = df[df.NAME_EN == 'Serbia'].iloc[0]
    serbia.ISO_A2 = 'CS'
    df = df.append(serbia)
    # French overseas territories are missing, merge with closest neighboring country
    # Guadeloupe, Martinique
    dominica = df[df.NAME_EN == 'Dominica'].iloc[0]
    for overseas_terriroty in ['MQ', 'GP']:
        dominica.ISO_A2 = overseas_terriroty
        df = df.append(dominica)
    # La Réunion, Mayotte
    madagascar = df[df.NAME_EN == 'Madagascar'].iloc[0]
    for overseas_terriroty in ['RE', 'YT']:
        madagascar.ISO_A2 = overseas_terriroty
        df = df.append(madagascar)
    # French Guyana
    suriname = df[df.NAME_EN == 'Suriname'].iloc[0]
    suriname.ISO_A2 = 'GF'
    df = df.append(suriname)
    # drop all other states that have an ivalid ISO code
    df = df[df['ISO_A2'] != '-99']
    return df
