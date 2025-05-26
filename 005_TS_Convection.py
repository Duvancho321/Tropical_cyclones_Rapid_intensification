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
from pandas import read_csv, to_datetime, DataFrame,concat
from matplotlib import pyplot as plt
from numpy import nan
from matplotlib.gridspec import GridSpec

#%% RI Treshold
valores_comparacion = {
    '12': 20 }

# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/TS/'
patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'
path_df = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'

#%% TC SELECTED
hur_sel = 'AL062022/'
name = 'EARL'.title()
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_df+hur_sel).glob(f"*Convective_C13*")])
paths = DataFrame({
    'path':dir_select
    })
paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-1], format='%Y%m%d%H%M.csv')
#paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-2], format='e%Y%j%H%M%S%f').dt.floor('min') + to_timedelta(1, unit='m')
paths.sort_values('date_time',ascending=True,inplace=True)

#%% DATA TRAYECTORIES
trayectories = read_csv(patha_data + 'trayectories12.csv',index_col=[0],parse_dates=[0])
trayectories['BASIN'] = trayectories['ATCF_ID'].str[0:2]
trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'EP', 'TLON']
trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON'] = -trayectories.loc[trayectories['BASIN'] == 'AL', 'TLON']
#%% FULL DATA CONVECTION
convection_data = concat([read_csv(file,parse_dates= ['fecha'],index_col=0) for file in paths['path'].values], ignore_index=True).replace(0,nan)
convection_data.set_index('fecha',inplace=True)

#%% PERIODOS DE INTENSIFICACION RAPIDA
trayectories_merge = trayectories[trayectories['ATCF_ID'] == hur_sel[0:-1]].reset_index(drop=False)
rapid = trayectories_merge[['date_time','VMAX']].reset_index(drop=False)
for obsi,lag in zip(valores_comparacion.keys(),[2]):
    rapid[f'RI'] = rapid['VMAX'].sub(rapid['VMAX'].shift(lag)).gt(valores_comparacion[obsi])
    rapid[f'RIP'] = rapid[f'RI'].shift(-lag)

rapid['IDX_RI'] = rapid['RI'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['IDX_RIP'] = rapid['RIP'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['periods'] = (rapid['IDX_RI'].fillna(0) + rapid['IDX_RIP'].fillna(0)).replace(0,nan)
periods_rapid_min = rapid.groupby('IDX_RIP')['date_time'].min().to_frame().rename(columns={'date_time':'min'})
periods_rapid_max = rapid.groupby('IDX_RI')['date_time'].max().to_frame().rename(columns={'date_time':'max'})
periods_rapid = concat([periods_rapid_min,periods_rapid_max],axis=1).sort_index().iloc[1:,:]

#%% GRAFICO
fig = plt.figure(figsize=(10.1,7.5),facecolor='white',edgecolor='white')
gs = GridSpec(3,1,wspace=.4,hspace=.3)
ax1,ax2,ax7 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0])
ax3,ax4 = ax1.twinx(),ax2.twinx()
ax5,ax6 = ax1.twinx(),ax2.twinx()
ax8,ax9 = ax7.twinx(),ax7.twinx()
ax5.spines.right.set_position(("axes", 1.1))
ax6.spines.right.set_position(("axes", 1.1))
ax9.spines.right.set_position(("axes", 1.1))
ax1.tick_params(axis='y', colors='#1f77b4',labelsize=7.5)
ax3.tick_params(axis='y', colors='#2b5c9d',labelsize=7.5)
ax5.tick_params(axis='y', colors='#2ca02c',labelsize=7.5)
ax2.tick_params(axis='y', colors='#1f77b4',labelsize=7.5)
ax4.tick_params(axis='y', colors='#2b5c9d',labelsize=7.5)
ax6.tick_params(axis='y', colors='#2ca02c',labelsize=7.5)
ax7.tick_params(axis='y', colors='#4361ee',labelsize=7.5)
ax8.tick_params(axis='y', colors='#540b0e',labelsize=7.5)
ax9.tick_params(axis='y', colors='#e09f3e',labelsize=7.5)
ax1.set(ylabel="Number of Structures")
ax3.set(ylabel="Number of Pixel [x10²]")
ax5.set(ylabel="Center Distance Std [Km]")
ax2.set(ylabel="Number of Structures")
ax4.set(ylabel="Number of Pixel [x10³]")
ax6.set(ylabel="Center Distance Std [Km]")
ax7.set(ylabel="Number of Hot Towers")
ax8.set(ylabel="Convective Systems\nArea [Km² x 10⁴]")
ax9.set(ylabel="Center Distance Std [Km]")
ax1.yaxis.label.set_color('#1f77b4')
ax3.yaxis.label.set_color('#2b5c9d')
ax5.yaxis.label.set_color('#2ca02c')
ax2.yaxis.label.set_color('#1f77b4')
ax4.yaxis.label.set_color('#2b5c9d')
ax6.yaxis.label.set_color('#2ca02c')
ax7.yaxis.label.set_color('#4361ee')
ax8.yaxis.label.set_color('#540b0e')
ax9.yaxis.label.set_color('#e09f3e')
for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['right'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))

ax1.set_title('Hot Towers',color = (0.45, 0.45, 0.45), pad=5, fontsize=9, fontweight='bold')
ax2.set_title('Convective Systems',color = (0.45, 0.45, 0.45), pad=5, fontsize=9, fontweight='bold')
fig.suptitle(f"{name} - {hur_sel[0:-1]}",
             #color = colores[hur_sel[0:-1]],
             color = (0.45, 0.45, 0.45),
             fontsize=12, fontweight='bold')
ax1.axvline(trayectories_merge['date_time'][trayectories_merge['VMAX'].idxmax()],
            color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
ax2.axvline(trayectories_merge['date_time'][trayectories_merge['VMAX'].idxmax()],
            color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
ax7.axvline(trayectories_merge['date_time'][trayectories_merge['VMAX'].idxmax()],
            color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
ax1.plot(convection_data['ht_count'],color = '#1f77b4',alpha=.6)
ax2.plot(convection_data['rb_count'],color = '#1f77b4',alpha=.6)
ax3.plot(convection_data['ht_pixel_count']/100,color = '#2b5c9d',alpha=.6)
ax4.plot(convection_data['rb_pixel_count']/1000,color = '#2b5c9d',alpha=.6)
ax5.plot(convection_data['ht_dist_std']/1000,color = '#2ca02c',alpha=.6)
ax6.plot(convection_data['rb_dist_std']/1000,color = '#2ca02c',alpha=.6)
ax7.plot(convection_data['ht_count'],color = '#4361ee',alpha=.8)
ax8.plot(convection_data['rb_area']/10_000,color = '#540b0e',alpha=.8)
ax9.plot(convection_data['rb_dist_std']/1000,color = '#e09f3e',alpha=.8)
for index,row in periods_rapid.iterrows():
    ax1.fill_betweenx([0,convection_data['ht_count'].max()],row['min'], row['max'], color='firebrick',alpha=0.15)
    ax2.fill_betweenx([0,convection_data['rb_count'].max()],row['min'], row['max'], color='firebrick',alpha=0.15)
    ax7.fill_betweenx([0,convection_data['ht_count'].max()],row['min'], row['max'], color='firebrick',alpha=0.15)
plt.savefig(path_fig+f"{hur_sel[0:-1]}_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.close()
# %%
