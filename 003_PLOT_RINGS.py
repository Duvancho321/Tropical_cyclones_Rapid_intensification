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
from pandas import read_csv, to_datetime, DataFrame,concat,date_range
from gc import collect
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from numpy import nan,where, meshgrid,arange,linspace
from fuction_tools import loadCPT
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import DateFormatter,HourLocator,DayLocator
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

#%% FUNCIONES
def smooth_data(data, window, iterations):
    for _ in range(iterations):
        data = data.rolling(window=window, center=True).mean()
    return data

#%% Diccionarios
names = {
    "WP262016": "MEARI",
    "WP062021": "CHAMPI",
    "EP112019": "JULIETTE",
    "WP202019": "HAGIBIS",
    "WP262019": "FENGSHEN",
    "EP052017": "EUGENE",
    "EP022019": "BARBARA",
    "WP032018": "JELAWAT",
    "WP342018": "MAN-YI",
    "WP222019": "BUALOI",
    "AL062019": "ERIN",
    "AL082018": "HELENE",
    "EP162020": "KARINA",
    "EP182021": "TERRY",
    "WP132017": "NALGAE",
    "WP192018": "LEEPI",
    "WP232015": "CHOI-WAN",
    "EP122021": "LINDA",
    "EP212018": "SERGIO",
    "WP102018": "MARIA",
    "WP102019": "LEKIMA",
    'WP162020': 'CHAN-HOM',
    'EP192020': 'NORBERT',
    'WP272017': 'SAOLA',
    'WP272015': 'IN-FA',
    'WP022016': 'NEPARTAK ',
    'WP022021': 'SURIGAE',
    'WP252017': 'LAN',
    'WP262018': 'MANGKHUT',
    'WP132015': 'SOUDELOR',
    'WP162016': 'MERANTI',
    'WP092015': 'CHAN-HOM',
    'WP142018': 'WUKONG',
    'WP172018': 'SHANSHAN',
    'EP022022': 'BLAS',
    'EP072019': 'FLOSSIE',
    'EP072022': 'FRANK',
    'EP082021': 'HILDA',
    'EP152018': 'MIRIAM'

}

colores = {'WP202019':'#540b0e','WP102018':'#9e2a2b','WP262019':'#e09f3e','EP112019':'#ffd60a',
           'WP222019':'#1a472a','EP022019':'#2d8659','WP032018':'#77ac30','WP342018':'#d4e157',
           'WP262016':'#3a0ca3','WP232015':'#7209b7','WP062021':'#4361ee','WP132017':'#4cc9f0',
           }
valores_comparacion = {
    '12': 20 }
# %% PATHS
path_fig = '/home/dunievesr/Datos/TESIS/FIG/'
path_df_ring = '/home/dunievesr/Datos/TESIS/DF_DATA/'

#%% PALETA
cpt = loadCPT('IR4AVHRR6.cpt')
# Makes a linear interpolation with the CPT file
cpt_convert = LinearSegmentedColormap('cpt', cpt)
lev_bt = linspace(170,378,200)
lev_dis = linspace(0,80,100) #pendiente de evaluar si el 80 cubre todos los maximos en todos los tC
norm_bt = BoundaryNorm(boundaries=lev_bt, ncolors = 256)
norm_dis = BoundaryNorm(boundaries=lev_dis, ncolors = 256)

#%% HURACAN DE PRUEBA
hur_sel = 'WP232015/'
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_df_ring+hur_sel).glob(f"*Rings_C13*")])
paths = DataFrame({
    'path':dir_select
    })
paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-1], format='%Y%m%d%H%M.csv')
#paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-2], format='e%Y%j%H%M%S%f').dt.floor('min') + to_timedelta(1, unit='m')
paths.sort_values('date_time',ascending=True,inplace=True)

#%% LECTURA DE TRAYECTORIA
trayectories = read_csv('/home/dunievesr/Dropbox/UNAL/TESIS/trayectories_interpolate.csv',index_col=0,parse_dates=[1])
trayectories = trayectories[trayectories['ATCF_ID']==hur_sel[0:-1]]
trayectories_merge = trayectories.merge(paths,how='left',on='date_time')
trayectories_merge ['bestrack'] = (trayectories_merge[trayectories.select_dtypes(float).dropna(axis=1).columns]
                                   .astype(str).apply(lambda x: x.str.split('.').str[1].str.len() < 2)).any(axis=1)
trayectories_merge.dropna(subset='path',inplace=True)
trayectories_merge.reset_index(inplace=True)

#%% PERIODOS DE INTENSIFICACION RAPIDA
rapid = trayectories_merge[trayectories_merge['bestrack']][['date_time','VMAX']].reset_index(drop=True)
for obsi,lag in zip(valores_comparacion.keys(),[4]):
    rapid[f'RI'] = rapid['VMAX'].sub(rapid['VMAX'].shift(lag)).gt(valores_comparacion[obsi])
    rapid[f'RIP'] = rapid[f'RI'].shift(-lag)

rapid['IDX_RI'] = rapid['RI'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['IDX_RIP'] = rapid['RIP'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['periods'] = (rapid['IDX_RI'].fillna(0) + rapid['IDX_RIP'].fillna(0)).replace(0,nan)
periods_rapid_min = rapid.groupby('IDX_RIP')['date_time'].min().to_frame().rename(columns={'date_time':'min'})
periods_rapid_max = rapid.groupby('IDX_RI')['date_time'].max().to_frame().rename(columns={'date_time':'max'})
periods_rapid = concat([periods_rapid_min,periods_rapid_max],axis=1).sort_index().iloc[1:,:]
#%%RESHAPE 
vars = ['min', 'max', 'mean', 'median', 'std', 'q1','q5','q10', 'q25', 'q50', 'q75', 'q90', 'q99', 'iqr']
all_files = concat([read_csv(file,parse_dates= ['date_time']) for file in trayectories_merge['path'].values], ignore_index=True)
reshape_data = all_files.pivot_table(
    values=vars,
    index=['date_time'],
    columns=['radial_interval'])
reshape_data.sort_index(inplace=True)
#reshape_data.to_csv(path_df_ring+'Rings_C13_'+hur_sel[0:-1]+'.csv')

# %% STD WP232015
var =  'std'
camp = 'jet' if var in ['iqr','std'] else cpt_convert
vmax = 80 if var == 'iqr' else 45
norm =  norm_dis if var in ['iqr','std'] else norm_bt
lev =  lev_dis if var in ['iqr','std'] else lev_bt
X, Y = meshgrid(arange(len(reshape_data[var].columns) + 1),
                arange(len(reshape_data[var].index) + 1))
Z = reshape_data[var]
fig = plt.figure(figsize=(4, 6),facecolor='w',edgecolor='w')
ax=fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color((0.45, 0.45, 0.45))
ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.plot(trayectories_merge['USA_RMW']/1000,trayectories_merge.index,color='black',alpha=.6,label='Inner Core')
ax.axhline(y=trayectories_merge.loc[trayectories_merge['VMAX'].idxmax()].name, color='#e09f3e', linestyle='-',lw=3, label='Max Intensity')
#ax.axvline(x=24, color='black', linestyle=':',lw=1.5)
if camp == 'jet':
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,vmax=vmax)
    colorbar = fig.colorbar(colormesh)
else:
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,norm=norm,alpha=.75)
    colorbar = fig.colorbar(colormesh,ticks=linspace(lev[0],lev[-1],15))
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
dates = reshape_data[var].index
unique_days = dates.floor('D').unique()
ax.set_yticks([where(dates.floor('D') == day)[0][0]+.5 for day in unique_days])
ax.set_yticklabels(unique_days.strftime('%Y-%m-%d'))    
ax.set_xticks(linspace(0,250,11))
ax.set_xticklabels(arange(0, 1100, 100).astype(int))
colorbar.ax.yaxis.label.set_color((0.45, 0.45, 0.45))
colorbar.ax.yaxis.label.set_size(14) 
ax.set_xlabel('Radii [Km]', fontsize=10, color=(0.45, 0.45, 0.45))
ax.set_ylim(Y[:,0][0],Y[:,0][-1])
colorbar.set_label('Brightness Temperature [K]', size=10,color=(0.45, 0.45, 0.45)) 
ax.grid(linestyle=':')
leg = ax.legend(fontsize = 10,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.07),
            frameon=False, ncol=3)
for text in leg.get_texts():
    text.set_color((0.45, 0.45, 0.45))
ax.set_title(f"Typical NI Case\n{var.upper()}",
                color = "#e09f3e", 
                #color = (0.45, 0.45, 0.45),
                pad=20, fontsize=12, fontweight='bold')
for index,row in periods_rapid.iterrows():
        ax.fill_between([0,250],
                        trayectories_merge[trayectories_merge['date_time']==row['min']].index.item(),
                        trayectories_merge[trayectories_merge['date_time']==row['max']].index.item(), 
                        color='black',alpha=0.25)
ax.text(-0.3, 1.135, f'(d)', 
        transform=ax.transAxes,
        color = (0.45, 0.45, 0.45),
        fontsize=12, fontweight='bold', va='top', ha='left')
plt.savefig(path_fig+f"Typical_NI_{var.upper()}",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Typical_NI_{var.upper()}.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.close()

# %% Q10 WP232015
var =  'q10'
camp = 'jet' if var in ['iqr','std'] else cpt_convert
vmax = 80 if var == 'iqr' else 45
norm =  norm_dis if var in ['iqr','std'] else norm_bt
lev =  lev_dis if var in ['iqr','std'] else lev_bt
X, Y = meshgrid(arange(len(reshape_data[var].columns) + 1),
                arange(len(reshape_data[var].index) + 1))
Z = reshape_data[var]
fig = plt.figure(figsize=(4, 6),facecolor='w',edgecolor='w')
ax=fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color((0.45, 0.45, 0.45))
ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.plot(trayectories_merge['USA_RMW']/1000,trayectories_merge.index,color='black',alpha=.6,label='Inner Core')
ax.axhline(y=trayectories_merge.loc[trayectories_merge['VMAX'].idxmax()].name, color='#e09f3e', linestyle='-',lw=3, label='Max Intensity')
#ax.axvline(x=140, color='black', linestyle=':',lw=1.5)
if camp == 'jet':
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,vmax=vmax)
    colorbar = fig.colorbar(colormesh)
else:
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,norm=norm,alpha=.75)
    colorbar = fig.colorbar(colormesh,ticks=linspace(lev[0],lev[-1],15))
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
dates = reshape_data[var].index
unique_days = dates.floor('D').unique()
ax.set_yticks([where(dates.floor('D') == day)[0][0]+.5 for day in unique_days])
ax.set_yticklabels(unique_days.strftime('%Y-%m-%d'))    
ax.set_xticks(linspace(0,250,11))
ax.set_xticklabels(arange(0, 1100, 100).astype(int))
colorbar.ax.yaxis.label.set_color((0.45, 0.45, 0.45))
colorbar.ax.yaxis.label.set_size(14) 
ax.set_xlabel('Radii [Km]', fontsize=10, color=(0.45, 0.45, 0.45))
ax.set_ylim(Y[:,0][0],Y[:,0][-1])
colorbar.set_label('Brightness Temperature [K]', size=10,color=(0.45, 0.45, 0.45)) 
ax.grid(linestyle=':')
leg = ax.legend(fontsize = 10,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.07),
            frameon=False, ncol=3)
for text in leg.get_texts():
    text.set_color((0.45, 0.45, 0.45))
ax.set_title(f"Typical NI Case\n{var.upper()}",
                color = "#e09f3e", 
                #color = (0.45, 0.45, 0.45),
                pad=20, fontsize=12, fontweight='bold')
for index,row in periods_rapid.iterrows():
        ax.fill_between([0,250],
                        trayectories_merge[trayectories_merge['date_time']==row['min']].index.item(),
                        trayectories_merge[trayectories_merge['date_time']==row['max']].index.item(), 
                        color='black',alpha=0.25)
ax.text(-0.3, 1.135, f'(b)', 
        transform=ax.transAxes,
        color = (0.45, 0.45, 0.45),
        fontsize=12, fontweight='bold', va='top', ha='left')
plt.savefig(path_fig+f"Typical_NI_{var.upper()}",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Typical_NI_{var.upper()}.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.close()

# %%
#%% HURACAN DE PRUEBA
#path_df_ring = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'
hur_sel = 'WP102018/'
band = 'B13' if hur_sel[0:2] == 'WP' else 'C13'
var = 'channel_0013_brightness_temperature' if hur_sel[0:2] == 'WP' else 'CMI'
dir_select =  sorted([str(x) for x in Path(path_df_ring+hur_sel).glob(f"*Rings_C13*")])
paths = DataFrame({
    'path':dir_select
    })
paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-1], format='%Y%m%d%H%M.csv')
#paths['date_time'] = to_datetime(paths['path'].str.split('/').str[-1].str.split('_').str[-2], format='e%Y%j%H%M%S%f').dt.floor('min') + to_timedelta(1, unit='m')
paths.sort_values('date_time',ascending=True,inplace=True)

#%% LECTURA DE TRAYECTORIA
trayectories = read_csv('/home/dunievesr/Dropbox/UNAL/TESIS/trayectories_interpolate.csv',index_col=0,parse_dates=[1])
trayectories = trayectories[trayectories['ATCF_ID']==hur_sel[0:-1]]
trayectories_merge = trayectories.merge(paths,how='left',on='date_time')
trayectories_merge ['bestrack'] = (trayectories_merge[trayectories.select_dtypes(float).dropna(axis=1).columns]
                                   .astype(str).apply(lambda x: x.str.split('.').str[1].str.len() < 2)).any(axis=1)
trayectories_merge.dropna(subset='path',inplace=True)
trayectories_merge.reset_index(inplace=True)

#%% PERIODOS DE INTENSIFICACION RAPIDA
rapid = trayectories_merge[trayectories_merge['bestrack']][['date_time','VMAX']].reset_index(drop=True)
for obsi,lag in zip(valores_comparacion.keys(),[2]):
    rapid[f'RI'] = rapid['VMAX'].sub(rapid['VMAX'].shift(lag)).gt(valores_comparacion[obsi])
    rapid[f'RIP'] = rapid[f'RI'].shift(-lag)

rapid['IDX_RI'] = rapid['RI'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['IDX_RIP'] = rapid['RIP'].astype('float').cumsum().drop_duplicates(keep='first')
rapid['periods'] = (rapid['IDX_RI'].fillna(0) + rapid['IDX_RIP'].fillna(0)).replace(0,nan)
periods_rapid_min = rapid.groupby('IDX_RIP')['date_time'].min().to_frame().rename(columns={'date_time':'min'})
periods_rapid_max = rapid.groupby('IDX_RI')['date_time'].max().to_frame().rename(columns={'date_time':'max'})
periods_rapid = concat([periods_rapid_min,periods_rapid_max],axis=1).sort_index().iloc[1:,:]
#%%RESHAPE 
vars = ['min', 'max', 'mean', 'median', 'std', 'q1','q5','q10', 'q25', 'q50', 'q75', 'q90', 'q99', 'iqr']
all_files = concat([read_csv(file,parse_dates= ['date_time']) for file in trayectories_merge['path'].values], ignore_index=True)
reshape_data = all_files.pivot_table(
    values=vars,
    index=['date_time'],
    columns=['radial_interval'])
reshape_data.sort_index(inplace=True)
#reshape_data.to_csv(path_df_ring+'Rings_C13_'+hur_sel[0:-1]+'.csv')

# %% STD WP102018
var =  'std'
camp = 'jet' if var in ['iqr','std'] else cpt_convert
vmax = 80 if var == 'iqr' else 45
norm =  norm_dis if var in ['iqr','std'] else norm_bt
lev =  lev_dis if var in ['iqr','std'] else lev_bt
X, Y = meshgrid(arange(len(reshape_data[var].columns) + 1),
                arange(len(reshape_data[var].index) + 1))
Z = reshape_data[var]
fig = plt.figure(figsize=(4, 6),facecolor='w',edgecolor='w')
ax=fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color((0.45, 0.45, 0.45))
ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.plot(trayectories_merge['USA_RMW']/1000,trayectories_merge.index,color='black',alpha=.6,label='Inner Core')
ax.axhline(y=trayectories_merge.loc[trayectories_merge['VMAX'].idxmax()].name, color='#4361ee', linestyle='-',lw=3, label='Max Intensity')
#ax.axvline(x=24, color='black', linestyle=':',lw=1.5)
if camp == 'jet':
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,vmax=vmax)
    colorbar = fig.colorbar(colormesh)
else:
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,norm=norm,alpha=.75)
    colorbar = fig.colorbar(colormesh,ticks=linspace(lev[0],lev[-1],15))
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
dates = reshape_data[var].index
unique_days = dates.floor('D').unique()
ax.set_yticks([where(dates.floor('D') == day)[0][0]+.5 for day in unique_days])
ax.set_yticklabels(unique_days.strftime('%Y-%m-%d'))    
ax.set_xticks(linspace(0,250,11))
ax.set_xticklabels(arange(0, 1100, 100).astype(int))
colorbar.ax.yaxis.label.set_color((0.45, 0.45, 0.45))
colorbar.ax.yaxis.label.set_size(14) 
ax.set_xlabel('Radii [Km]', fontsize=10, color=(0.45, 0.45, 0.45))
ax.set_ylim(Y[:,0][0],Y[:,0][-1])
colorbar.set_label('Brightness Temperature [K]', size=10,color=(0.45, 0.45, 0.45)) 
ax.grid(linestyle=':')
leg = ax.legend(fontsize = 10,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.07),
            frameon=False, ncol=3)
for text in leg.get_texts():
    text.set_color((0.45, 0.45, 0.45))
ax.set_title(f"Typical RI Case\n{var.upper()}",
                color = "#4361ee", 
                #color = (0.45, 0.45, 0.45),
                pad=20, fontsize=12, fontweight='bold')
for index,row in periods_rapid.iterrows():
        collection = ax.fill_between([0,250],
                        trayectories_merge[trayectories_merge['date_time']==row['min']].index.item(),
                        trayectories_merge[trayectories_merge['date_time']==row['max']].index.item(), 
                        color='black',alpha=0.4)
        collection.set_edgecolor('white')
        collection.set_linewidth(3)
ax.text(-0.3, 1.135, f'(c)', 
        transform=ax.transAxes,
        color = (0.45, 0.45, 0.45),
        fontsize=12, fontweight='bold', va='top', ha='left')
# Agregar flechas moradas
posiciones_flechas = [
    (120, 710, 40, 640),  # (x_inicio, y_inicio, x_fin, y_fin) para la primera flecha
    (120, 510, 40, 440),   # segunda flecha
    (120, 350, 50, 280),    # tercera flecha
    (120, 220, 40, 150)     # cuarta flecha
]

for x_inicio, y_inicio, x_fin, y_fin in posiciones_flechas:
    ax.annotate('', 
                xy=(x_fin, y_fin),        # punto final de la flecha
                xytext=(x_inicio, y_inicio),  # punto inicial de la flecha
                arrowprops=dict(
                    facecolor='purple',   # color de la flecha
                    edgecolor='none',     # sin borde
                    shrink=0.05,          # reducción del tamaño de la flecha
                    width=2,              # ancho de la flecha
                    headwidth=8,          # ancho de la punta de la flecha
                    alpha=0.8             # transparencia
                )
               )
plt.savefig(path_fig+f"Typical_RI_{var.upper()}",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Typical_RI_{var.upper()}.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.close()

# %% Q10 WP232015
var =  'q10'
camp = 'jet' if var in ['iqr','std'] else cpt_convert
vmax = 80 if var == 'iqr' else 45
norm =  norm_dis if var in ['iqr','std'] else norm_bt
lev =  lev_dis if var in ['iqr','std'] else lev_bt
X, Y = meshgrid(arange(len(reshape_data[var].columns) + 1),
                arange(len(reshape_data[var].index) + 1))
Z = reshape_data[var]
fig = plt.figure(figsize=(4, 6),facecolor='w',edgecolor='w')
ax=fig.add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color((0.45, 0.45, 0.45))
ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
ax.plot(trayectories_merge['USA_RMW']/1000,trayectories_merge.index,color='black',alpha=.6,label='Inner Core')
ax.axhline(y=trayectories_merge.loc[trayectories_merge['VMAX'].idxmax()].name, color='#4361ee', linestyle='-',lw=3, label='Max Intensity')
#ax.axvline(x=140, color='black', linestyle=':',lw=1.5)
if camp == 'jet':
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,vmax=vmax)
    colorbar = fig.colorbar(colormesh)
else:
    colormesh = ax.pcolormesh(X, Y, Z, cmap=camp,edgecolor=None,norm=norm,alpha=.75)
    colorbar = fig.colorbar(colormesh,ticks=linspace(lev[0],lev[-1],15))
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
dates = reshape_data[var].index
unique_days = dates.floor('D').unique()
ax.set_yticks([where(dates.floor('D') == day)[0][0]+.5 for day in unique_days])
ax.set_yticklabels(unique_days.strftime('%Y-%m-%d'))    
ax.set_xticks(linspace(0,250,11))
ax.set_xticklabels(arange(0, 1100, 100).astype(int))
colorbar.ax.yaxis.label.set_color((0.45, 0.45, 0.45))
colorbar.ax.yaxis.label.set_size(14) 
ax.set_xlabel('Radii [Km]', fontsize=10, color=(0.45, 0.45, 0.45))
ax.set_ylim(Y[:,0][0],Y[:,0][-1])
colorbar.set_label('Brightness Temperature [K]', size=10,color=(0.45, 0.45, 0.45)) 
ax.grid(linestyle=':')
leg = ax.legend(fontsize = 10,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.07),
            frameon=False, ncol=3)
for text in leg.get_texts():
    text.set_color((0.45, 0.45, 0.45))
ax.set_title(f"Typical RI Case\n{var.upper()}",
                color = "#4361ee", 
                #color = (0.45, 0.45, 0.45),
                pad=20, fontsize=12, fontweight='bold')
for index,row in periods_rapid.iterrows():
        collection = ax.fill_between([0,250],
                        trayectories_merge[trayectories_merge['date_time']==row['min']].index.item(),
                        trayectories_merge[trayectories_merge['date_time']==row['max']].index.item(), 
                        color='black',alpha=0.4)
        collection.set_edgecolor('white')
        collection.set_linewidth(3)
ax.text(-0.3, 1.135, f'(a)', 
        transform=ax.transAxes,
        color = (0.45, 0.45, 0.45),
        fontsize=12, fontweight='bold', va='top', ha='left')
ovalos = [
    (85, 620, 35, 80),  # (x_centro, y_centro, ancho, alto)
    (80, 490, 35, 80),
    (65, 340, 35, 80),
    (65, 195, 35, 80)
]

for x_centro, y_centro, ancho, alto in ovalos:
    ovalo = Ellipse((x_centro, y_centro), ancho, alto, fill=False, 
                    linestyle='--', edgecolor='black', linewidth=1.5, alpha=1)
    ax.add_patch(ovalo)
plt.savefig(path_fig+f"Typical_RI_{var.upper()}",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Typical_RI_{var.upper()}.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.close()
# %%
