#!/usr/bin/env python
# coding: utf-8

"""
Codigo para trabajo de analisis de datos, lectura de datos locales de GOES y sacar radio de interes.

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.4"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__refereces__:
    - [colors metpy](https://unidata.github.io/MetPy/latest/api/generated/metpy.plots.ctables.colortables.html#metpy.plots.ctables.colortables)
__methods__:
__comands__:
__changues__:
    - [2024-08-27][Duvan]: Primera version del codigo.

"""
#%% MODULOS
from pathlib import Path
from pandas import read_csv, to_datetime, DataFrame
from xarray import open_dataset, Dataset
from gc import collect

# %% PATHS
path_fig = '/home/dunievesr/Documents/UNAL/FIG/'
path_ring = '/var/data2/GOES_TRACK/Operational/RINGS/'
path_save = '/var/data2/GOES_TRACK/Operational/DF_DATA/'

#%% HURACAN DE PRUEBA
hur_sel = 'AL142024/'
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_ring+hur_sel).glob(f"*{band}*")])
dir_select2 =  sorted([str(x) for x in Path(path_save+hur_sel).glob(f"*{band}*")])
paths = DataFrame({
    'path':dir_select
    })
paths2 = DataFrame({
    'path':dir_select2
    })
if hur_sel[0:2] == 'WP':
    paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('-').str[0], format='%Y%m%d%H%M00').dt.floor('min')
    #paths2['date_time'] = to_datetime(paths2['path'].str.split('/').str[-1].str.split('-').str[0], format='%Y%m%d%H%M00').dt.floor('min')
else:
    paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-3], format='s%Y%j%H%M%S%f').dt.floor('min')
    #paths2['date_time'] = to_datetime(paths2['path'].str.split('/').str[-1].str.split('_').str[-3], format='s%Y%j%H%M%S%f').dt.floor('min')
paths.sort_values('date_time',ascending=True,inplace=True)
#paths2.sort_values('date_time',ascending=True,inplace=True)

#%% LECTURA DE DATOS
for i,row in paths.iterrows():
    print(len(paths)-i)
    #%%
    try:
        #%%
        with open_dataset(row['path']) as xds:
            data = xds[var]
            #%%
            stats = Dataset({
                'min': data.min(dim=['x', 'y'], skipna=True),
                'max': data.max(dim=['x', 'y'], skipna=True),
                'mean': data.mean(dim=['x', 'y'], skipna=True),
                'median': data.median(dim=['x', 'y'], skipna=True),
                'std': data.std(dim=['x', 'y'], skipna=True)
            })
            
            quantiles = data.chunk({'y': -1, 'x': -1}).quantile(
                [0.01,0.05,0.10, 0.25, 0.50, 0.75, 0.90, 0.99], 
                dim=['x', 'y'], 
                skipna=True)
            
            for q, label in zip([0.01,0.05,0.10, 0.25, 0.50, 0.75, 0.90, 0.99], 
                                ['q1','q5','q10', 'q25', 'q50', 'q75', 'q90', 'q99']):
                stats[label] = quantiles.sel(quantile=q)
            
            stats['iqr'] = stats['q75'] - stats['q25']
            stats_df = stats.to_dataframe()
            if hur_sel[0:2] == 'WP':
                stats_df.drop(columns=['time','quantile'],inplace=True)
            else:
                stats_df.drop(columns=['t','y_image','x_image','band_wavelength','band_id','quantile'],inplace=True)
            stats_df['date_time'] = row['date_time']
            #stats_df['bestrack'] = row['bestrack']
            stats_df.to_csv(path_save+hur_sel+'Rings_C13_'+row['date_time'].strftime('%Y%m%d%H%M')+'.csv')
    except Exception as e:
        print(f"Error {e} en fecha: {row['date_time']}")
        pass
collect()
print('terminado')


