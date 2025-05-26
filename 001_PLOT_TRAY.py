#!/usr/bin/env python
# coding: utf-8

"""
Codigo para trabajo de analisis de datos, lectura de datos locales de GOES.

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
    - [2024-09-05][Duvan]: Primera version del codigo.

"""
#%% MODULOS
from xarray import open_dataset
from pathlib import Path
from pandas import read_csv,date_range,DataFrame
from numpy import nan
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fuction_tools import loadCPT
from pyproj import CRS
import matplotlib
from matplotlib.gridspec import GridSpec
from gc import collect
from os import system,path
from fuction_tools import interpolate_trajectory
from shapely.geometry import Point,LineString
from shapely.ops import transform
from geopandas import GeoDataFrame
from pyproj import CRS
#matplotlib.use('Agg')

#%% PALETA
cpt = loadCPT('IR4AVHRR6.cpt')
# Makes a linear interpolation with the CPT file
cpt_convert = LinearSegmentedColormap('cpt', cpt)

# %% PATHS
path_fig = '/home/dunievesr/Datos/TESIS/FIG/'
patha_data = '/home/dunievesr/Dropbox/UNAL/TESIS/Final_codes/'

#%% DESCARGA DE ARCHIVO
#system(f'wget https://thredds.nci.org.au/thredds/fileServer/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/2019/10/18/1830/20191018183000-P1S-ABOM_OBS_B13-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')

#%% COLUMNAS INTERPOLATE
columns_to_interpolate = ['TLAT', 'TLON', 'VMAX', 'RSST', 'VMPI','SHRD','RHLO', 'USA_ROCI','USA_RMW','PSLV','NOHC','STORM_SPEED']

#%% HURACAN DE PRUEBA
hur_sel = 'WP252017/'
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
#%% TRAYECTORIA DE PRUEBA
trayectories = read_csv(patha_data + 'trayectories.csv',index_col=[0],parse_dates=[0])
trayectories['BASIN'] = trayectories['ATCF_ID'].str[0:2]
trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON']
trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON']
range_data = trayectories[trayectories['ATCF_ID']==hur_sel[0:-1]].sort_index()
interpolated_gdf = interpolate_trajectory(range_data, hur_sel[:-1],['TLAT', 'TLON', 'VMAX'])
interpolated_gdf['geometry'] = [Point(lon, lat) for lon, lat in zip(interpolated_gdf['TLON'], interpolated_gdf['TLAT'])]
interpolated_gdf = GeoDataFrame(interpolated_gdf, geometry='geometry', crs="EPSG:4326")
range_data['geometry'] = [Point(lon, lat) for lon, lat in zip(range_data['TLON'], range_data['TLAT'])]
range_data = GeoDataFrame(range_data, geometry='geometry', crs="EPSG:4326")
#%% CARGA DE DATOS
with open_dataset(patha_data+'20171022160000-P1S-ABOM_OBS_B13-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc') as xds:
    cc = CRS.from_cf(xds.geostationary.attrs)
    xds.rio.write_crs(cc.to_string(), inplace=True)
    sat_height = 1
    new_gdf = interpolated_gdf.to_crs(cc)
    line_gdf = GeoDataFrame(geometry=[LineString(list(new_gdf.geometry))], crs=new_gdf.crs)  
    x_min, y_min, x_max, y_max = new_gdf.total_bounds
    xds_cut = xds.sel(x=slice(x_min,x_max),y=slice(y_min,y_max))

    #%% FIGURA
    fig = plt.figure(figsize=(7,5.3),facecolor='None',edgecolor='None')
    ax = fig.add_subplot(111)
    img = xds_cut[var].T.plot(ax=ax, cmap=cpt_convert, vmin=170, vmax=378, add_colorbar=False)
    line_gdf['geometry'].apply(lambda geom: transform(lambda x, y: (y, x), geom)).plot(ax=ax,color='black')
    range_data.to_crs(cc)['geometry'].apply(lambda geom: transform(lambda x, y: (y, x), geom)).plot(ax=ax,color='black',alpha=.6)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.set_xticklabels([])  
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.setp(ax.spines.values(), color=(0.45, 0.45, 0.45))
    ax.set_title('')
    plt.savefig(path_fig+'trayectories.png',pad_inches=0.1,bbox_inches='tight',dpi=250)
    plt.close()
    

# %%
