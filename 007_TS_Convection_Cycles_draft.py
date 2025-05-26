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
from pandas import read_csv,DataFrame,concat,MultiIndex,cut
from matplotlib import pyplot as plt
from fuction_tools import restructure_hht_results,apply_fft,restructure_fft_results,integrate_sequential
from fuction_tools import valores_comparacion
from matplotlib.gridspec import GridSpec
from numpy import arange,pi,deg2rad,linspace,nanmean,isnan
from numpy.random import choice
from seaborn import histplot,kdeplot
from scipy.stats import mannwhitneyu

# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/'
path_df = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'
patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'

#%% DATOS CONVECTIVE
full_data = read_csv(path_df+'Convective_full_data.csv',index_col=[0])
full_data_scaled = read_csv(path_df+'Convective_full_data_scaled.csv',index_col=[0])
#full_data['hours_from_max'] = full_data['hours_from_max'].round(1)
summary_full_data = full_data_scaled.drop(columns='ATCF_ID').groupby(['RI','hours_from_max']).agg(['mean','std']).reset_index()
data_ni = summary_full_data[summary_full_data['RI']==0]
data_ri = summary_full_data[summary_full_data['RI']==1]

#%% DATOS TRAYECTORIES
rate_results = [] 
ships_select = read_csv(patha_data + 'trayectories12.csv',index_col=[0],parse_dates=[0])
for TC in full_data['ATCF_ID'].unique():
    rapid = ships_select[ships_select['ATCF_ID']==TC].sort_index()[['VMAX']].reset_index(drop=False)
    for obsi,lag in zip(valores_comparacion.keys(),[2]):
        rate_results.append({
            'ATCF_ID': TC,
            'rate': rapid['VMAX'].sub(rapid['VMAX'].shift(lag)).max(),
            'VMAX': rapid['VMAX'].max()
            })
df_rate = DataFrame(rate_results)
#%% VARIABLES SELECCIONADAS
variables = ['ht_pixel_count', 'rb_pixel_count', 'ht_count', 'rb_count', 'ht_area',
             'rb_area', 'ht_bt_mean', 'rb_bt_mean', 'ht_bt_median', 'rb_bt_median',
             'ht_bt_std', 'rb_bt_std', 'ht_bt_iqr', 'rb_bt_iqr', 'ht_dist_mean',
             'rb_dist_mean', 'ht_dist_median', 'rb_dist_median', 'ht_dist_std',
             'rb_dist_std', 'ht_dist_iqr', 'rb_dist_iqr']
variables = ["ht_count", "ht_area", "rb_bt_mean", 
             "rb_bt_std", "rb_bt_std", "rb_area"]

# %% DATTOS HHT
df_hht = restructure_hht_results(full_data,variables,repeat=False)
#%% DATOS FFT
df_fft = restructure_fft_results(full_data,variables)
bins = linspace(df_fft.index.min(),df_fft.index.max() , 601) 
df_fft = df_fft.groupby(
    cut(df_fft.index,
        bins=bins,
        labels=bins[:-1],
        include_lowest=True)
    ).max()

# %% RESUMEN DE RANGO DE FRECUENCIAS
dominant_freq = (df_hht.loc[:, df_hht.columns.get_level_values(3).str.contains('dominant')]
                 .T.droplevel('TC',axis=0)
                 .sort_index().mean(axis=1)
                 .groupby(level=[0,1,2])
                 .agg(['min','max']))

# %% GRAFICOS DE FRECUENCIAS DOMINANTES 
for var in dominant_freq.index.get_level_values(1).unique():
    var_range = dominant_freq.xs(var,level=1)
    y_labels = [f"IMF {item.split('_')[1]}" for item in var_range.xs('RI')['min'].index]
    
    fig = plt.figure(figsize=(6, 3), facecolor='w', edgecolor='w')
    ax = fig.add_subplot(111)
    
    # Crear posiciones Y ajustadas para cada serie
    y_positions = range(len(y_labels))
    y_positions_ri = [y + 0.15 for y in y_positions]
    y_positions_ni = [y - 0.15 for y in y_positions]
    
    # Dibujar líneas para RI
    ax.hlines(y=y_positions_ri, 
             xmin=var_range.xs('RI')['min'], 
             xmax=var_range.xs('RI')['max'], 
             color='#540b0e', 
             label='RI')
    ax.scatter(var_range.xs('RI')['min'], y_positions_ri, color='#540b0e', s=5)
    ax.scatter(var_range.xs('RI')['max'], y_positions_ri, color='#540b0e', s=5)
    
    # Dibujar líneas para NI
    ax.hlines(y=y_positions_ni, 
             xmin=var_range.xs('NI')['min'], 
             xmax=var_range.xs('NI')['max'], 
             color='#4361ee', 
             label='NI')
    ax.scatter(var_range.xs('NI')['min'], y_positions_ni, color='#4361ee', s=5)
    ax.scatter(var_range.xs('NI')['max'], y_positions_ni, color='#4361ee', s=5)
    
    # Configurar etiquetas del eje Y
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    
    # Agregar leyenda
    ax.invert_yaxis()

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    
    leg = fig.legend(bbox_to_anchor=(0.645, .99),frameon=False,
                    handlelength=2, ncol=2)

    for text in leg.get_texts():
        text.set_color((0.45, 0.45, 0.45))
        text.set_fontsize(8)
    ax.set_title(var.replace('_',' ').upper(), color=(0.45, 0.45, 0.45), pad=20, fontsize=12,fontweight='bold')
    ax.set_xlabel('Frequency', color=(0.45, 0.45, 0.45), fontsize=10)
    
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, color=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='minor', left=False)
    ax.grid(True,linestyle=':', color=(0.45, 0.45, 0.45), linewidth=0.5,alpha=.6)
    ax.grid(True, which='minor', axis='x', linestyle=':', color=(0.45, 0.45, 0.45), linewidth=0.5, alpha=0.2)

#%% RESAMPLE DE SERIE 
#bins = arange(-100, 200.1, 0.16666) 
#df_resampled = df_hht.groupby(
#    cut(df_hht.index,
#        bins=bins,
#        labels=bins[:-1],
#        include_lowest=True)
#    ).mean()
df_resampled = df_hht.copy()
# %% IMFs
IMFs = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('imf')]
# %% GRAFICOS IMFs
for var in IMFs.columns.get_level_values(2).unique():
    var_data = IMFs.xs(var,axis=1,level=2)
    fig, axs = plt.subplots(6, 2, figsize=(16, 15), facecolor='white', 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.4})
    axs = axs.flatten()
    for i, (ax, num) in enumerate(zip(axs, [i for i in range(7) for _ in range(2)])):
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color((0.45, 0.45, 0.45))
        ax.tick_params(axis='both', which='major', labelsize=7.5, 
                       colors=(0.45, 0.45, 0.45))

        if i % 2 == 0:
            data = var_data['RI'].xs(f'imf_{num+1}',axis=1,level=1)
            data.index = data.index.astype('float')
            data.plot(ax=ax,legend=False,color='#540b0e',alpha=.1)
            data.median(axis=1).plot(ax=ax,legend=False,color='#540b0e',alpha=.8)
            freq_min = dominant_freq.loc['RI',var,f'Freq_{num+1}_dominant']['min']
            freq_max = dominant_freq.loc['RI',var,f'Freq_{num+1}_dominant']['max']
            text = ax.text(.8, .95, f'Freq: {freq_min:.2f}-{freq_max:.2f}', color = (0.45, 0.45, 0.45),
                           transform=ax.transAxes, fontsize=7, weight='bold')
            ax.set_ylabel(f'IMF {num+1}', color=(0.45, 0.45, 0.45), fontsize=13)
        else:
            data = var_data['NI'].xs(f'imf_{num+1}',axis=1,level=1)
            data.index = data.index.astype('float')
            data.plot(ax=ax,legend=False,color='#4361ee',alpha=.1)
            data.median(axis=1).plot(ax=ax,legend=False,color='#4361ee',alpha=.8)
            freq_min = dominant_freq.loc['NI',var,f'Freq_{num+1}_dominant']['min']
            freq_max = dominant_freq.loc['NI',var,f'Freq_{num+1}_dominant']['max']
            text = ax.text(.8, .95, f'Freq: {freq_min:.2f}-{freq_max:.2f}', color = (0.45, 0.45, 0.45),
                           transform=ax.transAxes, fontsize=7, weight='bold')
        ax.set_xlabel('')
    fig.suptitle(var.replace('_',' ').upper(), fontsize=15, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% GRAFICOS FASE -IMF POLAR
IMF_PHA = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('imf|phase')]
for var in IMF_PHA.columns.get_level_values(2).unique():
    var_data = IMF_PHA.xs(var,axis=1,level=2)
    fig = plt.figure(figsize=(5, 5*6), facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(6, 1, i+1, projection='polar')
        ax.set_theta_zero_location("N")  
        ax.set_theta_direction(-1)       
        imfs = var_data.loc[:, var_data.columns.get_level_values(2).str.contains(f'imf_{i+1}')]
        #ax.set_rlim(imfs.min().min(), imfs.max().max())
        ax.grid(True, color=(0.45, 0.45, 0.45), alpha=0.3)
        ax.tick_params(colors=(0.45, 0.45, 0.45))
        ax.tick_params(labelcolor=(0.45, 0.45, 0.45))
        ax.spines['polar'].set_color((0.45, 0.45, 0.45))
        #ax.set_rgrids([2.2, 4.4, 6.6, 8.8, 11.0], angle=0)
        ax.set_thetagrids(arange(0, 360, 45))        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        all_imfs = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'imf_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ri = DataFrame({
            'imf': all_imfs['value'],
            'phase': all_phases['value']}).dropna().round(1)
        mean_ri = data_ri.groupby('phase').mean()
        std_ri = data_ri.groupby('phase').std()
        ax.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri.values, color='#540b0e', alpha=0.5)
        ax.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri-std_ri).values[:,0],(mean_ri+std_ri).values[:,0], color='#540b0e', alpha=0.1)
        case_data_ni = var_data.xs('NI',axis=1,level=0)
        all_imfs = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'imf_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ni = DataFrame({
            'imf': all_imfs['value'],
            'phase': all_phases['value']}).dropna().round(1)
        mean_ni = data_ni.groupby('phase').mean()
        std_ni = data_ni.groupby('phase').std()
        ax.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni.values, color='#4361ee', alpha=0.5)
        ax.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni-std_ni).values[:,0],(mean_ni+std_ni).values[:,0], color='#4361ee', alpha=0.1)
        ax.set_ylabel(f'IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10,labelpad=30)
        #ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% ESPECTRUM 1
ESPEC = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('freq|power')]
# %% GRAFICOS ESPECTROS 1
for var in ESPEC.columns.get_level_values(2).unique():
    var_data = ESPEC.xs(var,axis=1,level=2)
    fig, axs = plt.subplots(6, 1, figsize=(8, 13), facecolor='white', 
                           gridspec_kw={'wspace': 0.1, 'hspace': 0.4})
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        power_i = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains(f'power_{i+1}')]]
        max_power = power_i.max().max()
        min_power = power_i.min().min()
        max_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['max'].max()
        min_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['min'].min()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color((0.45, 0.45, 0.45))
        ax.tick_params(axis='both', which='major', labelsize=7.5, 
                      colors=(0.45, 0.45, 0.45))
        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        for TC in case_data_ri.columns.get_level_values(0):
            data_i = case_data_ri.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')].sort_values(by=f'freq_{i+1}')
                ax.vlines(data_i_freq[f'freq_{i+1}'],ymin=0,ymax=((data_i_freq[f'power_{i+1}']-min_power)/(max_power-min_power)),
                         alpha=.02,color='#540b0e')
            except:
                pass

        case_data_ni = var_data.xs('NI',axis=1,level=0)
        for TC in case_data_ni.columns.get_level_values(0):
            data_i = case_data_ni.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')].sort_values(by=f'freq_{i+1}')
                ax.vlines(data_i_freq[f'freq_{i+1}'],ymin=0,ymax=((data_i_freq[f'power_{i+1}']-min_power)/(max_power-min_power)),
                         alpha=.02,color='#4361ee')
            except:
                pass
        ax.set_ylabel(f'IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10)
        ax.set_xlim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% FASE - POWER POLAR
POW_PHA = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('power|phase')]
# %% GRAFICOS FASE - POWER POLAR
for var in POW_PHA.columns.get_level_values(2).unique():
    var_data = POW_PHA.xs(var,axis=1,level=2)
    fig = plt.figure(figsize=(5, 5*6), facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(6, 1, i+1, projection='polar')
        ax.set_theta_zero_location("N")  
        ax.set_theta_direction(-1)       
        powers = var_data.loc[:, var_data.columns.get_level_values(2).str.contains(f'power_{i+1}')]
        max_power = powers.max().max()
        min_power = powers.min().min()
        #ax.set_rlim(powers.min().min(), freqs.max().max())
        ax.grid(True, color=(0.45, 0.45, 0.45), alpha=0.3)
        ax.tick_params(colors=(0.45, 0.45, 0.45))
        ax.tick_params(labelcolor=(0.45, 0.45, 0.45))
        ax.spines['polar'].set_color((0.45, 0.45, 0.45))
        #ax.set_rgrids([2.2, 4.4, 6.6, 8.8, 11.0], angle=0)
        ax.set_thetagrids(arange(0, 360, 45))        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        all_powers = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'power_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ri = DataFrame({
            'power': (all_powers['value']-min_power)/(max_power-min_power),
            'phase': all_phases['value']}).dropna().round(1)
        mean_ri = data_ri.groupby('phase').mean()
        std_ri = data_ri.groupby('phase').std()
        ax.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri.values, color='#540b0e', alpha=0.5)
        ax.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri-std_ri).values[:,0],(mean_ri+std_ri).values[:,0], color='#540b0e', alpha=0.1)
        case_data_ni = var_data.xs('NI',axis=1,level=0)
        all_powers = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'power_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ni = DataFrame({
            'power': (all_powers['value']-min_power)/(max_power-min_power),
            'phase': all_phases['value']}).dropna().round(1)
        mean_ni = data_ni.groupby('phase').mean()
        std_ni = data_ni.groupby('phase').std()
        ax.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni.values, color='#4361ee', alpha=0.5)
        ax.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni-std_ni).values[:,0],(mean_ni+std_ni).values[:,0], color='#4361ee', alpha=0.1)
        ax.set_ylabel(f'IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10,labelpad=30)
        #ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 


# %% FRECUENCIA
FREQ = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('freq')]
# %% GRAFICOS FRECUENCIA
for var in FREQ.columns.get_level_values(2).unique():
    var_data = FREQ.xs(var,axis=1,level=2)
    fig, axs = plt.subplots(6, 1, figsize=(8, 13), facecolor='white', 
                           gridspec_kw={'wspace': 0.1, 'hspace': 0.4})
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        max_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['max'].max()
        min_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['min'].min()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color((0.45, 0.45, 0.45))
        ax.tick_params(axis='both', which='major', labelsize=7.5, 
                      colors=(0.45, 0.45, 0.45))
        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        for TC in case_data_ri.columns.get_level_values(0):
            data_i = case_data_ri.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')]
                ax.vlines(data_i_freq.index,ymin=0,ymax=data_i_freq[f'freq_{i+1}'],alpha=.02,color='#540b0e')
            except:
                pass

        case_data_ni = var_data.xs('NI',axis=1,level=0)
        for TC in case_data_ni.columns.get_level_values(0):
            data_i = case_data_ni.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')]
                ax.vlines(data_i_freq.index,ymin=0,ymax=data_i_freq[f'freq_{i+1}'],alpha=.02,color='#4361ee')
            except:
                pass
        ax.set_ylabel(f'FREQ IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10)
        ax.set_ylim(0)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 
# %% FASE
PHA = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('phase')]
# %% GRAFICOS FRECUENCIA- FASE
for var in PHA.columns.get_level_values(2).unique():
    var_data = PHA.xs(var,axis=1,level=2)
    fig, axs = plt.subplots(6, 1, figsize=(8, 13), facecolor='white', 
                           gridspec_kw={'wspace': 0.1, 'hspace': 0.4})
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        max_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['max'].max()
        min_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['min'].min()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color((0.45, 0.45, 0.45))
        ax.tick_params(axis='both', which='major', labelsize=7.5, 
                      colors=(0.45, 0.45, 0.45))
        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        median_ri = case_data_ri.droplevel(0,axis=1).loc[:,case_data_ri.droplevel(0,axis=1).columns.str.contains(f'{i+1}')].median(axis=1)
        for TC in case_data_ri.columns.get_level_values(0):
            data_i = case_data_ri.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')]#.sort_values(by=f'phase_{i+1}')
                ax.plot(data_i_freq.index,data_i_freq[f'phase_{i+1}'],alpha=.01,color='#540b0e')
                ax.plot(median_ri,alpha=.08,color='#540b0e')
            except:
                pass

        case_data_ni = var_data.xs('NI',axis=1,level=0)
        median_ni = case_data_ni.droplevel(0,axis=1).loc[:,case_data_ni.droplevel(0,axis=1).columns.str.contains(f'{i+1}')].median(axis=1)
        for TC in case_data_ni.columns.get_level_values(0):
            data_i = case_data_ni.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')].sort_values(by=f'phase_{i+1}')
                ax.plot(data_i_freq.index,data_i_freq[f'phase_{i+1}'],alpha=.01,color='#4361ee')
                ax.plot(median_ni,alpha=.08,color='#4361ee')
            except:
                pass
        ax.set_ylabel(f'PHASE IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10)
        #ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% GRAFICOS FASE POLAR
for var in PHA.columns.get_level_values(2).unique():
    var_data = PHA.xs(var,axis=1,level=2)
    fig = plt.figure(figsize=(5, 5*6), facecolor='white')
    for i in range(6):
        max_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['max'].max()
        min_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['min'].min()
        ax = fig.add_subplot(6, 1, i+1, projection='polar')
        ax.set_theta_zero_location("N")  
        ax.set_theta_direction(-1)       
        ax.set_rlim(-100, 0)
        ax.grid(True, color='gray', alpha=0.3)
        #ax.set_rgrids([2.2, 4.4, 6.6, 8.8, 11.0], angle=0)
        ax.set_thetagrids(arange(0, 360, 45))        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        median_ri = case_data_ri.droplevel(0,axis=1).loc[:,case_data_ri.droplevel(0,axis=1).columns.str.contains(f'{i+1}')]
        median_ri = median_ri.reset_index().melt(id_vars=['hours_from_max'],value_vars=median_ri.columns).round(1) #ROUND2
        median_ri = median_ri.select_dtypes('float').groupby('value').quantile(.90)
        median_ri.index = median_ri.index  * 180 / pi
        ax.plot(deg2rad(median_ri.index ), median_ri.values, color='#540b0e', alpha=0.5)
        for TC in case_data_ri.columns.get_level_values(0):
            data_i = case_data_ri.xs(TC,axis=1,level=0)
            try:
                phase_deg = data_i[f'phase_{i+1}']  * 180 / pi
                ax.plot(deg2rad(phase_deg), phase_deg.index.values, color='#540b0e', alpha=0.01)
            except:
                pass

        case_data_ni = var_data.xs('NI',axis=1,level=0)
        median_ni = case_data_ni.droplevel(0,axis=1).loc[:,case_data_ni.droplevel(0,axis=1).columns.str.contains(f'{i+1}')]
        median_ni = median_ni.reset_index().melt(id_vars=['hours_from_max'],value_vars=median_ni.columns).round(1) #ROUND2
        median_ni = median_ni.select_dtypes('float').groupby('value').quantile(.90)
        median_ni.index = median_ni.index  * 180 / pi
        ax.plot(deg2rad(median_ni.index ), median_ni.values, color='#4361ee', alpha=0.5)
        for TC in case_data_ni.columns.get_level_values(0):
            data_i = case_data_ni.xs(TC,axis=1,level=0)
            try:
                phase_deg = data_i[f'phase_{i+1}']  * 180 / pi
                ax.plot(deg2rad(phase_deg), phase_deg.index.values, color='#4361ee', alpha=0.01)
            except:
                pass
        ax.set_ylabel(f'IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10,labelpad=30)
        #ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% FRECUENCIA- FASE
FREQ_PHA = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('freq|phase')]
# %% GRAFICOS FRECUENCIA- FASE
for var in FREQ_PHA.columns.get_level_values(2).unique():
    var_data = FREQ_PHA.xs(var,axis=1,level=2)
    fig, axs = plt.subplots(6, 1, figsize=(8, 13), facecolor='white', 
                           gridspec_kw={'wspace': 0.1, 'hspace': 0.4})
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        max_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['max'].max()
        min_freq = dominant_freq.loc[:,var,f'Freq_{i+1}_dominant']['min'].min()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color((0.45, 0.45, 0.45))
        ax.tick_params(axis='both', which='major', labelsize=7.5, 
                      colors=(0.45, 0.45, 0.45))
        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        for TC in case_data_ri.columns.get_level_values(0):
            data_i = case_data_ri.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')].sort_values(by=f'phase_{i+1}')
                ax.plot(data_i_freq[f'phase_{i+1}'],data_i_freq[f'freq_{i+1}'],alpha=.02,color='#540b0e')
            except:
                pass

        case_data_ni = var_data.xs('NI',axis=1,level=0)
        for TC in case_data_ni.columns.get_level_values(0):
            data_i = case_data_ni.xs(TC,axis=1,level=0)
            try:
                data_i_freq = data_i.loc[:,data_i.columns.str.contains(f'{i+1}')].sort_values(by=f'phase_{i+1}')
                ax.plot(data_i_freq[f'phase_{i+1}'],data_i_freq[f'freq_{i+1}'],alpha=.02,color='#4361ee')
            except:
                pass
        ax.set_ylabel(f'FREQ IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10)
        ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 

# %% GRAFICOS FASE - FRECUENCIA POLAR
for var in FREQ_PHA.columns.get_level_values(2).unique():
    var_data = FREQ_PHA.xs(var,axis=1,level=2)
    fig = plt.figure(figsize=(5, 5*6), facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(6, 1, i+1, projection='polar')
        ax.set_theta_zero_location("N")  
        ax.set_theta_direction(-1)       
        freqs = var_data.loc[:, var_data.columns.get_level_values(2).str.contains(f'freq_{i+1}')]
        #ax.set_rlim(freqs.min().min(), freqs.max().max())
        ax.grid(True, color=(0.45, 0.45, 0.45), alpha=0.3)
        ax.tick_params(colors=(0.45, 0.45, 0.45))
        ax.tick_params(labelcolor=(0.45, 0.45, 0.45))
        ax.spines['polar'].set_color((0.45, 0.45, 0.45))
        #ax.set_rgrids([2.2, 4.4, 6.6, 8.8, 11.0], angle=0)
        ax.set_thetagrids(arange(0, 360, 45))        
        case_data_ri = var_data.xs('RI',axis=1,level=0)
        all_freqs = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'freq_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ri = DataFrame({
            'freq': all_freqs['value'],
            'phase': all_phases['value']}).dropna().round(2)
        mean_ri = data_ri.groupby('phase').mean()
        std_ri = data_ri.groupby('phase').std()
        ax.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri.values, color='#540b0e', alpha=0.5)
        ax.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri-std_ri).values[:,0],(mean_ri+std_ri).values[:,0], color='#540b0e', alpha=0.1)
        case_data_ni = var_data.xs('NI',axis=1,level=0)
        all_freqs = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'_{i+1}')].droplevel(0,axis=1).melt()
        all_phases = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains(f'phase_{i+1}')].droplevel(0,axis=1).melt()
        data_ni = DataFrame({
            'freq': all_freqs['value'],
            'phase': all_phases['value']}).dropna().round(2)
        mean_ni = data_ni.groupby('phase').mean()
        std_ni = data_ni.groupby('phase').std()
        ax.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni.values, color='#4361ee', alpha=0.5)
        ax.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni-std_ni).values[:,0],(mean_ni+std_ni).values[:,0], color='#4361ee', alpha=0.1)
        ax.set_ylabel(f'IMF {i+1}', color=(0.45, 0.45, 0.45), fontsize=10,labelpad=30)
        #ax.set_ylim(0,max_freq+0.0001)
    fig.suptitle(var.replace('_',' ').upper(), fontsize=13, y=0.9,fontweight='bold',color=(0.45, 0.45, 0.45)) 


# %%
fig = plt.figure(figsize=(4.75*4, 4.75*3), facecolor='white',edgecolor='none')
gs = GridSpec(3,4,wspace=.25)
ax1,ax4,ax7 = (fig.add_subplot(gs[0,0:2]),
               fig.add_subplot(gs[0,2],projection='polar'),
               fig.add_subplot(gs[0,3],projection='polar'))
ax2,ax5,ax8 = (fig.add_subplot(gs[1,0:2]),
               fig.add_subplot(gs[1,2],projection='polar'),
               fig.add_subplot(gs[1,3],projection='polar'))
ax3,ax6,ax9 = (fig.add_subplot(gs[2,0:2]),
               fig.add_subplot(gs[2,2],projection='polar'),
               fig.add_subplot(gs[2,3],projection='polar'))
for idx,ax in enumerate([ax1,ax2,ax3]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=10, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=10, colors=(0.45, 0.45, 0.45))
for idx,ax in enumerate([ax4,ax5,ax6,ax7,ax8,ax9]):
    ax.set_theta_zero_location("N")  
    ax.set_theta_direction(-1)
    ax.grid(True, color=(0.45, 0.45, 0.45), alpha=0.3)
    ax.tick_params(colors=(0.45, 0.45, 0.45))
    ax.tick_params(labelcolor=(0.45, 0.45, 0.45))
    ax.spines['polar'].set_color((0.45, 0.45, 0.45))
    ax.set_thetagrids(arange(0, 360, 45))      

for idx,var in enumerate(['ht_count','ht_area','rb_area']):
    var_data = df_resampled.xs(var,axis=1,level=2)
    var_data = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains('4')]]
    power = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains('power')]]
    IMFs = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains('imf')]]
    max_power = power.max().max()
    min_power = power.min().min()

    case_data_ri = var_data.xs('RI',axis=1,level=0)
    ims_ri = IMFs.xs('RI',axis=1,level=0)
    all_powers = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains('power')].droplevel(0,axis=1).melt()
    all_phases = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains('phase')].droplevel(0,axis=1).melt()
    all_imfs = case_data_ri.loc[:, case_data_ri.columns.get_level_values(1).str.contains('imf')].droplevel(0,axis=1).melt()
    data_ri = DataFrame({
        'imf':all_imfs['value'],
        'power': (all_powers['value']-min_power)/(max_power-min_power),
        'phase': all_phases['value']}).round(1)
    mean_ri = data_ri.groupby('phase').mean()
    std_ri = data_ri.groupby('phase').std()

    case_data_ni = var_data.xs('NI',axis=1,level=0)
    ims_ni = IMFs.xs('NI',axis=1,level=0)
    all_powers = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains('power')].droplevel(0,axis=1).melt()
    all_phases = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains('phase')].droplevel(0,axis=1).melt()
    all_imfs = case_data_ni.loc[:, case_data_ni.columns.get_level_values(1).str.contains('imf')].droplevel(0,axis=1).melt()
    data_ni = DataFrame({
        'imf':all_imfs['value'],
        'power': (all_powers['value']-min_power)/(max_power-min_power),
        'phase': all_phases['value']}).round(1)
    mean_ni = data_ni.groupby('phase').mean()
    std_ni = data_ni.groupby('phase').std()
    if idx == 0:
        ims_ri.plot(ax=ax1,legend=False,color='#540b0e',alpha=.1)
        ims_ni.plot(ax=ax1,legend=False,color='#4361ee',alpha=.1)
        ims_ri.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax1,legend=False,color='#540b0e',alpha=.8)
        ims_ni.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax1,legend=False,color='#4361ee',alpha=.8)
        ax4.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['imf'].values, color='#540b0e', alpha=0.5)
        ax4.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['imf']-std_ni['imf']).values,(mean_ri['imf']+std_ni['imf']).values, color='#540b0e', alpha=0.1)
        ax4.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['imf'].values, color='#4361ee', alpha=0.5)
        ax4.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['imf']-std_ni['imf']).values,(mean_ni['imf']+std_ni['imf']).values, color='#4361ee', alpha=0.1)
        ax7.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['power'].values, color='#540b0e', alpha=0.5)
        ax7.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['power']-std_ni['power']).values,(mean_ri['power']+std_ni['power']).values, color='#540b0e', alpha=0.1)
        ax7.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['power'].values, color='#4361ee', alpha=0.5)
        ax7.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['power']-std_ni['power']).values,(mean_ni['power']+std_ni['power']).values, color='#4361ee', alpha=0.1)
    if idx == 1:
        ims_ri.plot(ax=ax2,legend=False,color='#540b0e',alpha=.1)
        ims_ni.plot(ax=ax2,legend=False,color='#4361ee',alpha=.1)
        ims_ri.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax2,legend=False,color='#540b0e',alpha=.8)
        ims_ni.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax2,legend=False,color='#4361ee',alpha=.8)
        ax5.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['imf'].values, color='#540b0e', alpha=0.5)
        ax5.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['imf']-std_ni['imf']).values,(mean_ri['imf']+std_ni['imf']).values, color='#540b0e', alpha=0.1)
        ax5.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['imf'].values, color='#4361ee', alpha=0.5)
        ax5.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['imf']-std_ni['imf']).values,(mean_ni['imf']+std_ni['imf']).values, color='#4361ee', alpha=0.1)
        ax8.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['power'].values, color='#540b0e', alpha=0.5)
        ax8.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['power']-std_ni['power']).values,(mean_ri['power']+std_ni['power']).values, color='#540b0e', alpha=0.1)
        ax8.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['power'].values, color='#4361ee', alpha=0.5)
        ax8.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['power']-std_ni['power']).values,(mean_ni['power']+std_ni['power']).values, color='#4361ee', alpha=0.1)
    if idx == 2:
        ims_ri.plot(ax=ax3,legend=False,color='#540b0e',alpha=.1)
        ims_ni.plot(ax=ax3,legend=False,color='#4361ee',alpha=.1)
        ims_ri.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax3,legend=False,color='#540b0e',alpha=.8)
        ims_ni.median(axis=1).rolling(window=12, center=True).mean().plot(ax=ax3,legend=False,color='#4361ee',alpha=.8)
        ax6.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['imf'].values, color='#540b0e', alpha=0.5)
        ax6.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['imf']-std_ni['imf']).values,(mean_ri['imf']+std_ni['imf']).values, color='#540b0e', alpha=0.1)
        ax6.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['imf'].values, color='#4361ee', alpha=0.5)
        ax6.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['imf']-std_ni['imf']).values,(mean_ni['imf']+std_ni['imf']).values, color='#4361ee', alpha=0.1)
        ax9.plot(deg2rad(mean_ri.index* 180 / pi), mean_ri['power'].values, color='#540b0e', alpha=0.5)
        ax9.fill_between(deg2rad(mean_ri.index* 180 / pi), (mean_ri['power']-std_ni['power']).values,(mean_ri['power']+std_ni['power']).values, color='#540b0e', alpha=0.1)
        ax9.plot(deg2rad(mean_ni.index* 180 / pi), mean_ni['power'].values, color='#4361ee', alpha=0.5)
        ax9.fill_between(deg2rad(mean_ni.index* 180 / pi), (mean_ni['power']-std_ni['power']).values,(mean_ni['power']+std_ni['power']).values, color='#4361ee', alpha=0.1)

ax1.set_ylim(-65,65)
ax2.set_ylim(-1500,1500)
ax3.set_ylim(-120000,120000)
ax1.set_xlabel('[Hours]',color=(0.45, 0.45, 0.45), fontsize=12)
ax2.set_xlabel('[Hours]',color=(0.45, 0.45, 0.45), fontsize=12)
ax3.set_xlabel('[Hours]',color=(0.45, 0.45, 0.45), fontsize=12)
ax1.set_ylabel('Number of Hot Towers',color=(0.45, 0.45, 0.45), fontsize=15)
ax2.set_ylabel('Area of Hot Towers',color=(0.45, 0.45, 0.45), fontsize=15)
ax3.set_ylabel('Area of of Convective Systems',color=(0.45, 0.45, 0.45), fontsize=15)
ax1.set_title("Time series",color=(0.45, 0.45, 0.45), pad=5, fontsize=15, fontweight='bold')
ax4.set_title("IMFs",color=(0.45, 0.45, 0.45), pad=5, fontsize=15, fontweight='bold')
ax7.set_title("Power",color=(0.45, 0.45, 0.45), pad=5, fontsize=15, fontweight='bold')
fig.suptitle('Empirical Mode 5 (Freq: 0 - 0.01)', fontsize=19, y=.94,fontweight='bold',color=(0.45, 0.45, 0.45))
#plt.savefig(path_fig+f"IMF5_Cycles.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
#plt.savefig(path_fig+f"IMF5_Cycles.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)  
# %% FOURIER
dfs = []
for idx,var in enumerate(['ht_count','ht_area','rb_area']):
    var_data = df_resampled.xs(var,axis=1,level=2)
    IMFs = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains('imf_5')]]
    for TC in IMFs.columns.get_level_values(1):
        case_i = IMFs.xs(TC,axis=1,level=1).columns.get_level_values(0)[0]
        TC_IMF = IMFs.xs(TC,axis=1,level=1).droplevel(0,axis=1)
        TC_IMF = TC_IMF - TC_IMF.mean()
        fourier, freq, periodo, potencia  = apply_fft(TC_IMF.values)
        if case_i not in data:
            data[case_i] = {}
        df = DataFrame({
            'fourier': fourier[:,0],
            'period': periodo,
            'power': potencia[:,0]
        })
        df.columns = MultiIndex.from_product([[case_i], [TC],[var], df.columns], names=['case', 'TC','var', 'metric'])
        df.index = freq
        dfs.append(df)

fourier = concat(dfs,axis=1)

#%% ESPECTRO DE FOURIER 
pos_fourier = df_fft[df_fft.index.astype('float')>0]
pos_fourier.index = pos_fourier.index.astype('float')
fig, axs = plt.subplots(5, 1, figsize=(9, 15), facecolor='white', 
                        gridspec_kw={'wspace': 0.1, 'hspace': 0.3})
axs = axs.flatten()
for i,(var,ax) in enumerate(zip(pos_fourier.columns.get_level_values(2).unique(),axs)):
    var_data = pos_fourier.xs(var,axis=1,level=2)
    power_i = var_data.loc[:, var_data.columns[var_data.columns.get_level_values(2).str.contains('power')]]
    max_power = power_i.max().max()
    min_power = power_i.min().min()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                    colors=(0.45, 0.45, 0.45))
    case_data_ri = power_i.xs('RI',axis=1,level=0)
    for TC in case_data_ri.columns.get_level_values(0):
        data_i = case_data_ri.xs(TC,axis=1,level=0)
        ax.vlines(data_i.index,ymin=0,ymax=((data_i-min_power)/(max_power-min_power)),alpha=.1,color='#540b0e')
        #ax.vlines(data_i.index,ymin=0,ymax=data_i,alpha=.1,color='#540b0e')
        #ax.plot(data_i.index,data_i,alpha=.1,color='#540b0e')
        #ax.plot(data_i.index,((data_i-min_power)/(max_power-min_power)),alpha=.1,color='#540b0e')

    case_data_ni = power_i.xs('NI',axis=1,level=0)
    for TC in case_data_ni.columns.get_level_values(0):
        data_i = case_data_ni.xs(TC,axis=1,level=0)
        ax.vlines(data_i.index,ymin=0,ymax=((data_i-min_power)/(max_power-min_power)),alpha=.1,color='#4361ee')
        #ax.vlines(data_i.index,ymin=0,ymax=data_i,alpha=.1,color='#4361ee')
        #ax.plot(data_i.index,data_i,alpha=.1,color='#4361ee')
        #ax.plot(data_i.index,((data_i-min_power)/(max_power-min_power)),alpha=.1,color='#4361ee')
    ax.set_ylabel(f'Power', color=(0.45, 0.45, 0.45), fontsize=10)
    #ax.set_yscale('log',base=10)
    #ax.set_xscale('log',base=10)
    ax.set_xlim(0,.16666)
    ax.set_title(var.replace('_',' ').upper(), fontsize=13, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45)) 


#%% INTEGRAL 6h
bootstrap = True
df_fft_pos = df_fft[df_fft.index.astype('float')>0]
df_fft_pos.index = 1/(6*df_fft_pos.index.astype('float'))
integrales1 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[2,12,24,48,135])
integrales2 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[2,18,36,135])
integrales3 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[2,24,60])
integrales4 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[2,36])
integrales5 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[2,6,12,48])
integrales6 = integrate_sequential(df_fft_pos.sort_index().xs('power',axis=1,level=3),[3,18,26,50])
integrales_total = concat([integrales1,integrales2,integrales3,integrales4,integrales5,integrales6],axis=1).T
for var in integrales_total.columns.get_level_values(2).unique():
    var_data = integrales_total.xs(var,axis=1,level=2)
    for lead_time in var_data.index:
        if var in ['ht_area','ht_count','rb_area'] and lead_time in ['3-18','20-30']:
            lead_data = var_data.loc[lead_time]
            fig = plt.figure(figsize=(6, 3), facecolor='w', edgecolor='w')
            ax = fig.add_subplot(111)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color((0.45, 0.45, 0.45))
            ax.tick_params(axis='both', which='major', labelsize=7.5,colors=(0.45, 0.45, 0.45))
            if bootstrap:
                bootstrap_ri = choice(lead_data['RI'].dropna(), size=(10_000, len(lead_data['RI'])), replace=True)
                bootstrap_ni = choice(lead_data['NI'].dropna(), size=(10_000, len(lead_data['NI'])), replace=True)
                histplot(x=bootstrap_ri.flatten(), color="#540b0e", kde=True,ax=ax,label=False,bins=9,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'})
                histplot(x=bootstrap_ni.flatten(), color="#4361ee", kde=True,ax=ax,label=False,bins=9,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'})
                ax.axvline(lead_data['RI'].mean(), color="#540b0e")
                ax.axvline(lead_data['NI'].mean(), color="#4361ee")
                ax.axvline(bootstrap_ri.flatten().mean(), color="#540b0e")
                ax.axvline(bootstrap_ni.flatten().mean(), color="#4361ee")
                t_stat, p_valor = mannwhitneyu(bootstrap_ri.flatten(), bootstrap_ni.flatten(), alternative='two-sided')
                print(f"Estadístico t: {t_stat}")
                print(f"Valor p: {p_valor}")

            else:
                histplot(x=lead_data['RI'].values, color="#540b0e", kde=True,ax=ax,label=False,edgecolor=(0.45, 0.45, 0.45),bins=9,line_kws={'linestyle': '--'})
                histplot(x=lead_data['NI'].values, color="#4361ee", kde=True,ax=ax,label=False,edgecolor=(0.45, 0.45, 0.45),bins=9,line_kws={'linestyle': '--'})
                ax.axvline(lead_data['RI'].mean(), color="#540b0e")
                ax.axvline(lead_data['NI'].mean(), color="#4361ee")
                t_stat, p_valor = mannwhitneyu(lead_data['RI'].dropna().values, lead_data['NI'].dropna().values, alternative='two-sided')
                print(f"Estadístico t: {t_stat}")
                print(f"Valor p: {p_valor}")
            ax.set_ylabel(f'Count', color=(0.45, 0.45, 0.45), fontsize=10)
            ax.set_title(var.replace('_',' ').upper() + ' ' +lead_time, fontsize=13, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))
            plt.savefig(path_fig+var+lead_time+f".png",pad_inches=0.1,bbox_inches='tight',dpi=250)   

    
        
#%% HIST
var = 'ht_area'
df_fft_pos = df_fft[df_fft.index.astype('float')>0]
df_fft_pos.index = 1/(6*df_fft_pos.index.astype('float'))
hist_data = df_fft_pos.xs((var, 'power'),axis=1,level=[2, 3]).loc[48:20]
ri_data =  hist_data['RI'].values.flatten()
ri_data = ri_data[~isnan(ri_data)]
ni_data =  hist_data['NI'].values.flatten()
ni_data = ni_data[~isnan(ni_data)]
fig = plt.figure(figsize=(6, 3), facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=7.5,colors=(0.45, 0.45, 0.45))
histplot(x=ri_data, color="#540b0e", kde=True,ax=ax,label=False,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'})
histplot(x=ni_data, color="#4361ee", kde=True,ax=ax,label=False,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'})
ax.set_ylabel(f'Count', color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_title(var.replace('_',' ').upper(), fontsize=13, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45)) 
ax.axvline(nanmean(ri_data), color="#540b0e")
ax.axvline(nanmean(ni_data), color="#4361ee")
t_stat, p_valor = mannwhitneyu(ri_data, ni_data, alternative='two-sided')
print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_valor}")


#%%
var = 'ht_area'
lead_time = '2-24'
bins = 4 if lead_time == '2-24' else 7
var_data = integrales_total.xs(var,axis=1,level=2)
lead_data = var_data.loc[lead_time]
fig = plt.figure(figsize=(4.3, 3), facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=7.5,colors=(0.45, 0.45, 0.45))
bootstrap_ri = choice(lead_data['RI'].dropna(), size=(1_000, len(lead_data['RI'])), replace=True)
bootstrap_ni = choice(lead_data['NI'].dropna(), size=(1_000, len(lead_data['NI'])), replace=True)
histplot(x=bootstrap_ri.flatten(), color="#540b0e", kde=False,ax=ax,label=False,
         bins=bins,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=bootstrap_ri.flatten(), color="#540b0e", ax=ax, linestyle='--',
        bw_adjust=5, common_norm=False) #adjust5,3
histplot(x=bootstrap_ni.flatten(), color="#4361ee", kde=False,ax=ax,label=False,
         bins=bins+2,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=bootstrap_ni.flatten(), color="#4361ee", ax=ax, linestyle='--',
        bw_adjust=5, common_norm=False) #adjust5,3
ax.axvline(lead_data['RI'].mean(), color="#540b0e")
ax.axvline(lead_data['NI'].mean(), color="#4361ee")
ax.axvline(bootstrap_ri.flatten().mean(), color="#540b0e")
ax.axvline(bootstrap_ni.flatten().mean(), color="#4361ee")
t_stat, p_valor = mannwhitneyu(bootstrap_ri.flatten(), bootstrap_ni.flatten(), alternative='two-sided')

ax.set_ylabel(f'Density', color=(0.45, 0.45, 0.45), fontsize=10)
for j, stat in enumerate(['Samples Boostrap','P-value']):
    y_pos = 0.95 - (j*0.05)
    val = 1_000 if j == 0 else p_valor
    if isinstance(val, (float)):
        val_text = f"{val:.4f}"
    else:
        val_text = int(val)
    text = ax.text(0.67, y_pos, f'{stat}: ', color = (0.45, 0.45, 0.45),#.02
                    transform=ax.transAxes, fontsize=6, weight='bold')

    ax.annotate(val_text, xycoords=text, xy=(1, 0),
                verticalalignment="bottom", fontsize=6,
                color=(0.45, 0.45, 0.45), style="italic")

#ax.set_title(var.replace('_',' ').upper() + ' ' +lead_time, fontsize=13, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))
ax.set_title('Power Distribution Area of\nHot Towers'+ ' ' +lead_time , fontsize=13, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))
plt.savefig(path_fig+var+lead_time+f".png",pad_inches=0.1,bbox_inches='tight',dpi=250)   



# %% EXPERIMENTO IMFS 2
IMF5 = df_resampled.loc[:, df_resampled.columns[df_resampled.columns.get_level_values(3).str.contains('imf_5')]].droplevel(level=3,axis=1)
IMF5 = IMF5.unstack().reset_index().pivot(columns='variable', index=['Case', 'TC', 'hours_from_max'])[0].reset_index()#.set_index('hours_from_max')
IMF5.rename(columns={"Case": "RI", "TC": "ATCF_ID"},inplace=True)
IMF5.loc[IMF5['RI'] == 'RI', 'RI'] = 1
IMF5.loc[IMF5['RI'] == 'NI', 'RI'] = 0
df_hht5 = restructure_hht_results(IMF5,['ht_count','ht_area','rb_area'],repeat=False)



# %% ESPECTRUM 1
var_imf = {}
data_list = []
POW = df_resampled.loc[:, df_resampled.columns.get_level_values(3).str.contains('imf')]
for var in POW.columns.get_level_values(2).unique():
    var_imf[var] = {}
    var_data = POW.xs(var,axis=1,level=2)
    for TC in var_data.columns.get_level_values(1):
        TC_data = var_data.xs(TC,axis=1,level=1).droplevel(axis=1,level=0).var()
        data = full_data.set_index('hours_from_max')
        data = data[data['ATCF_ID']==TC].loc[:,var][-100:0]
        var_imf[var][TC] = (TC_data /data.var())*100
        row_dict = {
            'case': var_data.xs(TC,axis=1,level=1).columns.get_level_values(0).unique()[0],
            'Variable': var,
            'TC': TC }
        
        for imf_name, value in TC_data.items():
            percentage = (value/data.var())*100
            row_dict[imf_name] = percentage

        data_list.append(row_dict)

# %%
df_final.query("Variable=='ht_area'")[['case','imf_5']].pivot(columns='case').plot()
# %%
