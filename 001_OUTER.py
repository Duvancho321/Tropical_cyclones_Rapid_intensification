#!/usr/bin/env python
# coding: utf-8

"""
Codigo para descarga de datos de Himawari y GOES potimizada con radio respecto al centro
__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.4"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__changues__:
    - [2024-12-27][Duvan]: Primera version del codigo.
"""
#%% MODULOS
import multiprocessing
from dask import config
from pandas import read_csv,date_range
from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client, LocalCluster
from fuction_tools import interpolate_trajectory,process_time_goes,process_time_himawari

#%% Config
multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    #% CONFIG
    multiprocessing.freeze_support()
    cluster = LocalCluster(n_workers=30, threads_per_worker=1, memory_limit='2GB')
    client = Client(cluster)

    config.set({
        'array.chunk-size': '100MB',
        'distributed.worker.memory.target': 0.8,
        'distributed.worker.memory.spill': 0.85,
        'distributed.worker.memory.pause': 0.95,
    })

    #%% PATHS
    path_raw_goes = '/home/dunievesr/Documents/UNAL/RAW_GOES/'
    path_raw_him = '/home/dunievesr/Documents/UNAL/RAW_HIM/'
    path_cut = '/var/data2/GOES_TRACK/FINAL/'
    patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'

    #% BANDS
    bands = [13]

    #% COLUMNS INTERPOLATE
    columns_to_interpolate = ['TLAT', 'TLON', 'VMAX', 'RSST', 'VMPI','SHRD','RHLO', 'USA_ROCI','USA_RMW','PSLV','NOHC','STORM_SPEED']

    #%% READ DATA
    trayectories = read_csv(patha_data + 'trayectories.csv',index_col=[0],parse_dates=[0])
    trayectories['BASIN'] = trayectories['ATCF_ID'].str[0:2]
    trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON']
    trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON']
    procesados = [
            'EP062022', 'EP112019', 'EP202018', 'EP212018', 'WP022016',
            'WP022021', 'WP032017', 'WP032018', 'WP062021', 'WP102018',
            'WP132015', 'WP142017', 'WP152020', 'WP162016', 'WP202016',
            'WP202019', 'WP222019', 'WP232020', 'WP252017', 'WP252021',
            'WP262016', 'WP262018', 'WP262019', 'WP272015', 'WP272017',
            'WP342018', 'AL082019'
            ]
    trayectories = trayectories[trayectories['ATCF_ID'].isin(['AL082019','AL062018','AL122017','EP122019','AL062022'])]
    #%% CICLON 
    for hur_sel in trayectories['ATCF_ID'].unique():
        freq = '10min' if hur_sel[0:2] == 'WP' else '1h'
        satellite = '16' if hur_sel[0:2] == 'AL' else '17' if hur_sel[0:2] == 'EP' else '8' if hur_sel[0:2] == 'WP' else None
        range_data = trayectories[trayectories['ATCF_ID']==hur_sel].sort_index()
        interpolated_gdf = interpolate_trajectory(range_data, hur_sel,columns_to_interpolate)
        range_time = date_range(range_data.index.min(),
                                range_data.index.max(), freq=freq)
        process_func = process_time_himawari if hur_sel[0:2] == 'WP' else process_time_goes
        raw_path = path_raw_him if hur_sel[0:2] == 'WP' else path_raw_goes
        print(hur_sel,len(interpolated_gdf))
        
        #%% POOL DOWNLOAD
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = list(executor.map(process_func,
                                     [date for date in range_time],
                                     [interpolated_gdf]*len(range_time),
                                     [raw_path]*len(range_time),
                                     [path_cut]*len(range_time),
                                     [bands]*len(range_time),
                                     [hur_sel]*len(range_time),
                                     [satellite]*len(range_time)
                                     ))

    client.close()

#%% 
import xarray as xr





# %%
