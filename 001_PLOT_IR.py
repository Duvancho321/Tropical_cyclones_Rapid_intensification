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
from pandas import DataFrame,to_datetime
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fuction_tools import loadCPT
from pyproj import CRS
import matplotlib
from matplotlib.gridspec import GridSpec
from gc import collect
matplotlib.use('Agg')

#%% PALETA
cpt = loadCPT('IR4AVHRR6.cpt')
# Makes a linear interpolation with the CPT file
cpt_convert = LinearSegmentedColormap('cpt', cpt)

# %% PATHS
path_fig = '/home/dunievesr/Documents/UNAL/Operative/FIG/'
path_outer = '/var/data2/GOES_TRACK/Operational/OUTER/'

#%% HURACAN DE PRUEBA
hur_sel = 'AL142024/'
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_outer+hur_sel[0:-1]).glob(f"*{band}*")])
paths = DataFrame({
    'path':dir_select
    })
if hur_sel[0:2] == 'WP':
    paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('-').str[0], format='%Y%m%d%H%M00').dt.floor('min')
else:
    paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-3], format='s%Y%j%H%M%S%f').dt.floor('min')

paths.sort_values('date_time',ascending=True,inplace=True)

#%% LECTURA DE DATOS
for i,row in paths.iterrows():
    print(len(paths)-i)
    #% LECTURA DE XARRAY
    try:
        with open_dataset(row['path']) as xds:
            if hur_sel[0:2] == 'WP':
                cc = CRS.from_cf(xds.geostationary.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = 1
            else:
                cc = CRS.from_cf(xds.goes_imager_projection.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
            #% FIGURA
            fig = plt.figure(figsize=(7,5.3),facecolor='white',edgecolor='white')
            gs = GridSpec(1,2,wspace=.4,hspace=.3,width_ratios=[.1,1])#,height_ratios=[1,1,1,0.05])#,width_ratios=[1,0.05])#,height_ratios=[1,1,0.05])
            ax = fig.add_subplot(gs[:,1:3])
            cax = fig.add_subplot(gs[:,0])
            img = xds[var].plot(ax=ax,cmap=cpt_convert,vmin=170, vmax=378,add_colorbar=False)
            cbar = plt.colorbar(img, cax=cax, orientation='vertical',fraction=0.3, aspect=100)
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
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(color=(0.45, 0.45, 0.45))
            cbar.ax.yaxis.set_label_position('left')
            cbar.set_label('Brightness Temperature [K]', fontsize=12, color=(0.45, 0.45, 0.45))
            [tick.set_color((0.45, 0.45, 0.45)) for tick in cbar.ax.get_yticklabels()]
            plt.setp(ax.spines.values(), color=(0.45, 0.45, 0.45))
            ax.set_title(f"MILTON - {hur_sel[:-1]}\n{row['date_time']} UTC",
                    color = (0.45, 0.45, 0.45), pad=10, fontsize=11, fontweight='bold')
            name = f"{str(i).zfill(4)}_{hur_sel[:-1]}_{row['date_time'].strftime('%Y%m%d%H%M')}_{band}"
            plt.savefig(path_fig+name,pad_inches=0.1,bbox_inches='tight',dpi=250)
            plt.close()
    except:
        pass
collect()
print('terminadoo')        
#%%        