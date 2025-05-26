#!/usr/bin/env python
# coding: utf-8

"""
Codigo para trabajo de analisis de datos, descarga de datos de GOES con recorte por reproject.

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.4"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__refereces__:
__methods__:
__comands__:
__changues__:
    - [2024-04-27][Duvan]: Primera version del codigo.
    - [2024-04-28][Duvan]: Se ajusta para ciclos.
    - [2024-08-01][Duvab]: Se hace reprojeccion de puntos y no del dataset, menor tiempo de procesamiento
"""
#%% MODULOS
import multiprocessing
from dask import config
from pathlib import Path
from dask.distributed import Client, LocalCluster
from pandas import read_csv,DataFrame,to_datetime
from concurrent.futures import ThreadPoolExecutor
from fuction_tools import interpolate_trajectory,process_time_convective

#%% Config
multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    
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
    path_cut = '/var/data2/GOES_TRACK/FINAL/'
    patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'

    #% COLUMNS INTERPOLATE
    columns_to_interpolate = ['TLAT', 'TLON', 'VMAX', 'RSST', 'VMPI','SHRD','RHLO', 'USA_ROCI','USA_RMW','PSLV','NOHC','STORM_SPEED']

    #%% READ DATA
    trayectories = read_csv(patha_data + 'trayectories.csv',index_col=[0],parse_dates=[0])
    trayectories['BASIN'] = trayectories['ATCF_ID'].str[0:2]
    trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON']
    trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON']
    procesados = [
            "WP232020", "WP142017", "WP202016", "WP252021", "EP212018",
            "WP272015", "WP022021", "WP152020", "WP032017", "WP022016",
            "WP272017", "WP342018", "EP202018", "WP262018", "WP132015",
            "WP032018", "WP102018", "WP202019", "WP222019", "WP062021",
            "WP162016", "WP252017", "WP262019", "EP062022", "WP262016"
    ]
    trayectories = trayectories[trayectories['ATCF_ID'].isin(['WP142017'])]

    #%% CICLON 
    for hur_sel in trayectories['ATCF_ID'].unique():
        band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
        var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
        range_data = trayectories[trayectories['ATCF_ID']==hur_sel].sort_index()
        interpolated_gdf = interpolate_trajectory(range_data, hur_sel,columns_to_interpolate)
        dir_select =  sorted([str(x) for x in Path(path_cut+'OUTER/'+f'{hur_sel}/').glob(f"*{band}*")])   
        paths = DataFrame({
        'path':dir_select
        })
        if hur_sel[0:2] == 'WP':
            paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('-').str[0], format='%Y%m%d%H%M00').dt.floor('min')
        else:
            paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-3], format='s%Y%j%H%M%S%f').dt.floor('min')
        paths.sort_values('date_time',ascending=True,inplace=True)
        full_data = interpolated_gdf.merge(paths,how='left',on='date_time').dropna(subset='path')
        num_processes = 15
        delay = 55  # seconds
        delay_list = [delay * (1 + i) for i in range(num_processes)]
        repeat_count = (len(full_data) // num_processes) + 1  
        delay_list = (delay_list * repeat_count)[:len(full_data)] 

        #%% POOL PROCESS
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = list(executor.map(process_time_convective,
                                        [row for _, row in full_data.iterrows()],
                                        delay_list,
                                        [path_cut]*len(full_data),
                                        [hur_sel]*len(full_data),
                                        [var]*len(full_data),
                                        ))

# %%
