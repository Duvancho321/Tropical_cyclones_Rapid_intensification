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
from numpy import nanmedian,nanpercentile,array
from scipy.ndimage import label
#matplotlib.use('Agg')

#%% PALETA
cpt = loadCPT('IR4AVHRR6.cpt')
# Makes a linear interpolation with the CPT file
cpt_convert = LinearSegmentedColormap('cpt', cpt)

# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/CONVECTION/'
path_outer = '/var/data2/GOES_TRACK/FINAL/OUTER/'
path_con = '/var/data2/GOES_TRACK/FINAL/CONVECTIVE/'
path_save = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'

#%% HURACAN DE PRUEBA
hur_sel = 'WP142017/'
nombre = 'BANYAN'.title()
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_con+hur_sel[:-1]).glob(f"*{band}*")])
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
    #%% LECTURA DE XARRAY
    try:
        #%%
        with open_dataset(row['path'],chunks='auto') as cxds, open_dataset(row['path'].replace(path_con,path_outer),chunks='auto') as xds:
            if hur_sel[0:2] == 'WP':
                cc = CRS.from_cf(xds.geostationary.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = 1
            else:
                cc = CRS.from_cf(xds.goes_imager_projection.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
            #%% ANALISIS DE ESTRUCTURAS CONVECTIVAS
            #Conteo pixeles
            ht_pixel_count = cxds['mask_ht'].sum().compute().item()
            rb_pixel_count = cxds['mask_rb'].sum().compute().item()
            #Conteo de carateriticas con condiones de BT
            _,ht_count = label(cxds['mask_ht'])
            _,rb_count = label(cxds['mask_rb'])          
            #ht_count = cxds['mask_ht'].num_features
            #rb_count = cxds['mask_rb'].num_features
            #area
            ht_area = ht_pixel_count*4
            rb_area = rb_pixel_count*4
            #temperatura de brillo
            ht_bt_mean = cxds['bt_ht'].mean().compute().item()
            rb_bt_mean = cxds['bt_rb'].mean().compute().item()
            ht_bt_median = nanmedian(cxds['bt_ht'].values)
            rb_bt_median = nanmedian(cxds['bt_rb'].values)
            ht_bt_std = cxds['bt_ht'].std().compute().item()
            rb_bt_std = cxds['bt_rb'].std().compute().item()
            ht_bt_iqr = nanpercentile(cxds['bt_ht'], 75) - nanpercentile(cxds['bt_ht'], 25)
            rb_bt_iqr = nanpercentile(cxds['bt_rb'], 75) - nanpercentile(cxds['bt_rb'], 25)
            #distancia
            ht_dist_mean = cxds['dist_ht'].mean().compute().item()
            rb_dist_mean = cxds['dist_rb'].mean().compute().item()
            ht_dist_median = nanmedian(cxds['dist_ht'].values)
            rb_dist_median = nanmedian(cxds['dist_rb'].values)
            ht_dist_std = cxds['dist_ht'].std().compute().item()
            rb_dist_std = cxds['dist_rb'].std().compute().item()
            ht_dist_iqr = nanpercentile(cxds['dist_ht'], 75) - nanpercentile(cxds['dist_ht'], 25)
            rb_dist_iqr = nanpercentile(cxds['dist_rb'], 75) - nanpercentile(cxds['dist_rb'], 25)
            #% DATOS PARA DATAFRAME Y GRAFICA
            DataFrame({
                'fecha':[row['date_time']],
                'ht_pixel_count': [ht_pixel_count],
                'rb_pixel_count': [rb_pixel_count],
                'ht_count': [ht_count],
                'rb_count': [rb_count],
                'ht_area': [ht_area], #km2
                'rb_area': [rb_area],
                'ht_bt_mean': [ht_bt_mean],
                'rb_bt_mean': [rb_bt_mean],
                'ht_bt_median': [ht_bt_median],
                'rb_bt_median': [rb_bt_median],
                'ht_bt_std': [ht_bt_std],
                'rb_bt_std': [rb_bt_std],
                'ht_bt_iqr': [ht_bt_iqr],
                'rb_bt_iqr': [rb_bt_iqr],
                'ht_dist_mean': [ht_dist_mean*sat_height], #metros
                'rb_dist_mean': [rb_dist_mean*sat_height],
                'ht_dist_median': [ht_dist_median*sat_height],
                'rb_dist_median': [rb_dist_median*sat_height],
                'ht_dist_std': [ht_dist_std*sat_height],
                'rb_dist_std': [rb_dist_std*sat_height],
                'ht_dist_iqr': [ht_dist_iqr*sat_height],
                'rb_dist_iqr': [rb_dist_iqr*sat_height]
            }).to_csv(path_save+hur_sel+'Convective_C13_'+row['date_time'].strftime('%Y%m%d%H%M')+'.csv')
            #%% PLOT
            fig = plt.figure(figsize=(10.8,5.6),facecolor='white',edgecolor='white')
            gs = GridSpec(2,3,wspace=.4,hspace=.3,width_ratios=[.1,1,.425])#,height_ratios=[1,1,1,0.05])#,width_ratios=[1,0.05])#,height_ratios=[1,1,0.05])
            ax1 = fig.add_subplot(gs[:,1:2])
            cax = fig.add_subplot(gs[:,0])
            ax2,ax3 = fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])
            img  = xds[var].plot(ax=ax1,cmap=cpt_convert,vmin=170, vmax=378,add_colorbar=False)
            img2 = xds[var].plot(ax=ax2,cmap=cpt_convert,vmin=170, vmax=378,add_colorbar=False,alpha=.2)
            img3 = xds[var].plot(ax=ax3,cmap=cpt_convert,vmin=170, vmax=378,add_colorbar=False,alpha=.2)
            for j,ax in enumerate([ax1,ax2,ax3]):
                alpha = .7 if j == 0  else .3
                linewidth = 2  if j == 0  else 1
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])  
                ax.set_yticks([]) 
                ax.set_xticklabels([])  
                ax.set_yticklabels([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #% TRAZOS BUFFER
            ax1.set_title(f"{nombre} - {hur_sel[:-1]}\n{row['date_time']} UTC",
                    color = (0.45, 0.45, 0.45), pad=10, fontsize=11, fontweight='bold')
            ax2.set_title(f"Hot Towers",
                    color = (0.45, 0.45, 0.45), pad=30, fontsize=11, fontweight='bold')
            ax2.text(0.5, 1.1, f"Mean distance: {round((ht_dist_mean*sat_height)/1000,1)}±{round((ht_dist_std*sat_height)/1000,1)} Km\n# Pixels:{ht_pixel_count} Area:{ht_area} Km²\nMean BT: {round(ht_bt_mean,1)}±{round(ht_bt_std,1)}", 
                     ha='center', va='center',color = (0.45, 0.45, 0.45),fontsize=7, transform=ax2.transAxes)
            ax3.set_title(f"Rainbands",
                    color = (0.45, 0.45, 0.45), pad=30, fontsize=11, fontweight='bold')
            ax3.text(0.5, 1.1, f"Mean distance: {round((rb_dist_mean*sat_height)/1000,1)}±{round((rb_dist_std*sat_height)/1000,1)} Km\n# Pixels:{rb_pixel_count} Area:{rb_area} Km²\nMean BT: {round(rb_bt_mean,1)}±{round(rb_bt_std,1)}", 
                     ha='center', va='center',color = (0.45, 0.45, 0.45),fontsize=7, transform=ax3.transAxes)  
            #% BARRA DE COLORES
            cbar = plt.colorbar(img, cax=cax, orientation='vertical',fraction=0.3, aspect=100)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(color=(0.45, 0.45, 0.45))
            cbar.ax.yaxis.set_label_position('left')
            cbar.set_label('Brightness Temperature [K]', fontsize=12, color=(0.45, 0.45, 0.45))
            [tick.set_color((0.45, 0.45, 0.45)) for tick in cbar.ax.get_yticklabels()]

            #% ESTRUCTURAS CONVECTIVAS
            ax2.contour(xds.x, xds.y, cxds['mask_ht'], colors=['#FF00FF'], linewidths=1)
            ax3.contour(xds.x, xds.y, cxds['mask_rb'], colors=['#d00000'], linewidths=.1)

            #% NOMBRE Y GUARDADO DE FIGURA
            name = f"{str(i).zfill(4)}_{hur_sel[:-1]}_{row['date_time'].strftime('%Y%m%d%H%M')}_{band}"
            plt.savefig(path_fig+name,pad_inches=0.1,bbox_inches='tight',dpi=250)
            plt.close()
            #%%
    except:
        pass
collect()
print('terminadoo')        
#%%        
