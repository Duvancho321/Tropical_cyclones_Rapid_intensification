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
from fuction_tools import asignar_colores,interpolate_trajectory,min_max_scaler
from fuction_tools import names,paleta_amarillo_verde_azul,paleta_cian_rojo,valores_comparacion
# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/'
path_df = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'
patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'

#%% DATOS
ships_select = read_csv(patha_data + 'trayectories12.csv',index_col=[0],parse_dates=[0])
ships_select['BASIN'] = ships_select['ATCF_ID'].str[0:2]
ships_select['paso'] = (ships_select.groupby('ATCF_ID').cumcount() * 6)
ships_select['max_intensity_date'] = ships_select.groupby('ATCF_ID')['VMAX'].transform(
    lambda x: x.index[x == x.max()][0])
ships_select['hours_from_max']  = (ships_select.index - ships_select['max_intensity_date']).dt.total_seconds() / 3600
ships_select.loc[ships_select['BASIN'] == 'EP', 'TLON'] = -ships_select.loc[ships_select['BASIN'] == 'EP', 'TLON']
ships_select.loc[ships_select['BASIN'] == 'AL', 'TLON'] = -ships_select.loc[ships_select['BASIN'] == 'AL', 'TLON']

#%% Asignación de colores
colores = ships_select.groupby(['ATCF_ID','FINAL_O_INTENSITY'])[['VMAX']].max().reset_index().sort_values(['FINAL_O_INTENSITY','VMAX']).reset_index(drop=True)
colores.loc[colores['FINAL_O_INTENSITY']=='RI', "color"] = asignar_colores(colores[colores['FINAL_O_INTENSITY']=='RI'].index, paleta_cian_rojo) 
colores.loc[colores['FINAL_O_INTENSITY']=='NI', "color"] = asignar_colores(colores[colores['FINAL_O_INTENSITY']=='NI'].index, paleta_amarillo_verde_azul) 
colores.set_index('ATCF_ID',inplace=True)


#%%
list_data0 = []
list_data = []
fig = plt.figure(figsize=(16,6.5),facecolor='white',edgecolor='white')
gs = GridSpec(2,3,wspace=.2,hspace=.4)
ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])
ax3,ax4 = fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])
ax5,ax6 = fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])
for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
    ax.axvline(0,color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
    ax.set_xlim(-6,6)
ax1.set_title('Hot Towers',color = (0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
ax3.set_title('Convective Systems Area',color = (0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
ax5.set_title('Center Distance Std',color = (0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
ax1.set_ylabel("Count",color = (0.45, 0.45, 0.45))
ax2.set_ylabel("Count",color = (0.45, 0.45, 0.45))
ax3.set_ylabel("Km² x 10⁴",color = (0.45, 0.45, 0.45))
ax4.set_ylabel("Km² x 10⁴",color = (0.45, 0.45, 0.45))
ax5.set_ylabel("Km",color = (0.45, 0.45, 0.45))
ax6.set_ylabel("Km",color = (0.45, 0.45, 0.45))

for TC in names.keys():
    #% HURACAN DE PRUEBA
    hur_sel = f'{TC}/'
    try:
        #% COLUMNS INTERPOLATE
        columns_to_interpolate = ['TLAT', 'TLON', 'VMAX', 'RSST', 'VMPI','SHRD','RHLO', 'USA_ROCI','USA_RMW','PSLV','NOHC','STORM_SPEED']
        #%
        var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
        dir_select =  sorted([str(x) for x in Path(path_df+hur_sel).glob(f"*Convective_C13*")])
        paths = DataFrame({
            'path':dir_select
            })
        paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-1], format='%Y%m%d%H%M.csv')
        #paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-2], format='e%Y%j%H%M%S%f').dt.floor('min') + to_timedelta(1, unit='m')
        paths.sort_values('date_time',ascending=True,inplace=True)

        #% LECTURA DE TRAYECTORIA
        range_data = ships_select[ships_select['ATCF_ID']==hur_sel[0:-1]].sort_index()
        interpolated_gdf = interpolate_trajectory(range_data, hur_sel[0:-1],columns_to_interpolate)

        trayectories_merge = interpolated_gdf[interpolated_gdf['ATCF_ID'] == hur_sel[0:-1]].reset_index(drop=True)
        trayectories_merge = trayectories_merge.merge(paths,how='left',on='date_time').dropna(subset='path')

        #%FULL DATA 
        convection_data0 = concat([read_csv(file,parse_dates= ['fecha'],index_col=0) for file in trayectories_merge['path'].values], ignore_index=True)
        columns_to_scale = [col for col in convection_data0.columns if col != 'fecha']
        convection_data = convection_data0.copy()
        convection_data[columns_to_scale] = convection_data[columns_to_scale].apply(min_max_scaler)
        max_intensity_date = range_data['VMAX'].idxmax()
        convection_data0['hours_from_max']  = (convection_data0['fecha'] - max_intensity_date).dt.total_seconds() / 3600
        convection_data['hours_from_max']  = (convection_data['fecha'] - max_intensity_date).dt.total_seconds() / 3600
        convection_data['ATCF_ID'] = TC;convection_data0['ATCF_ID'] = TC
        
        #% PERIODOS DE INTENSIFICACION RAPIDA
        rapid = range_data[['VMAX']].reset_index(drop=False)
        for obsi,lag in zip(valores_comparacion.keys(),[2]):
            rapid[f'RI'] = rapid['VMAX'].sub(rapid['VMAX'].shift(lag)).gt(valores_comparacion[obsi])
            rapid[f'RIP'] = rapid[f'RI'].shift(-lag)

        rapid['IDX_RI'] = rapid['RI'].astype('float').cumsum().drop_duplicates(keep='first')
        rapid['IDX_RIP'] = rapid['RIP'].astype('float').cumsum().drop_duplicates(keep='first')
        rapid['periods'] = (rapid['IDX_RI'].fillna(0) + rapid['IDX_RIP'].fillna(0)).replace(0,nan)
        periods_rapid_min = rapid.groupby('IDX_RIP')['date_time'].min().to_frame().rename(columns={'date_time':'min'})
        periods_rapid_max = rapid.groupby('IDX_RI')['date_time'].max().to_frame().rename(columns={'date_time':'max'})
        periods_rapid = concat([periods_rapid_min,periods_rapid_max],axis=1).sort_index().iloc[1:,:]
        #%
        if periods_rapid.empty:
            convection_data0['RI'] = nan
            convection_data['RI'] = nan
            ax2.plot(convection_data['hours_from_max'],convection_data['ht_count'],color = 'gray',alpha=.3)
            ax4.plot(convection_data['hours_from_max'],convection_data['rb_area']/10_000,color = 'gray',alpha=.3)
            ax6.plot(convection_data['hours_from_max'],convection_data['rb_dist_std']/1000,color = 'gray',alpha=.3)
        else:
            first_ri = periods_rapid.loc[1,'min']
            convection_data0['RI'] = convection_data0['fecha'] == first_ri
            convection_data['RI'] = convection_data['fecha'] == first_ri
            ax1.plot(convection_data['hours_from_max'],convection_data['ht_count'],color = 'gray',alpha=.3)
            ax3.plot(convection_data['hours_from_max'],convection_data['rb_area']/10_000,color = 'gray',alpha=.3)
            ax5.plot(convection_data['hours_from_max'],convection_data['rb_dist_std']/1000,color = 'gray',alpha=.3)
            ax1.axvline(convection_data[convection_data['RI']]['hours_from_max'].item(),color = 'gray',alpha=.3,linestyle=':')
            ax3.axvline(convection_data[convection_data['RI']]['hours_from_max'].item(),color = 'gray',alpha=.3,linestyle=':')
            ax5.axvline(convection_data[convection_data['RI']]['hours_from_max'].item(),color = 'gray',alpha=.3,linestyle=':')     
        list_data0.append(convection_data0)
        list_data.append(convection_data)
    except:
        pass
plt.savefig(path_fig+f"Full_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)      
#%% 
full_data0 = concat(list_data0)
full_data = concat(list_data)
full_data['RI'] = full_data['RI'].replace(0,1)
full_data['RI'] = full_data['RI'].fillna(0)
full_data0['RI'] = full_data0['RI'].replace(0,1)
full_data0['RI'] = full_data0['RI'].fillna(0)
full_data0.to_csv(path_df+'Convective_full_data.csv',index=False)
full_data.to_csv(path_df+'Convective_full_data_scaled.csv',index=False)
#%%