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
from pandas import read_csv,DataFrame,concat
from matplotlib import pyplot as plt
from fuction_tools import stats_trend,create_plots_ts,add_stats_ts,add_pie_ts
from fuction_tools import valores_comparacion
from seaborn import regplot
from matplotlib.gridspec import GridSpec

# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/'
path_df = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'
patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'

#%% DATOS CONVECTIVE
full_data = read_csv(path_df+'Convective_full_data.csv',index_col=[0])
full_data_scaled = read_csv(path_df+'Convective_full_data_scaled.csv',index_col=[0])
#full_data['hours_from_max'] = full_data['hours_from_max'].round(1)
summary_full_data = full_data#.drop(columns='ATCF_ID')#.groupby(['RI','hours_from_max']).agg(['mean','std']).reset_index()
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
variables = ["ht_count", "rb_count", "ht_dist_mean", "rb_dist_mean", 
             "rb_area", "rb_bt_std", "ht_dist_std","rb_dist_std"]

#%% ANALISIS DE TENDENCIA INDIVIDUAL
results = []
for var in variables:
    for TC in full_data['ATCF_ID'].unique():
        stats_tc = stats_trend(full_data[full_data['ATCF_ID'] == TC], var)
        stats_tc_s = stats_trend(full_data_scaled[full_data_scaled['ATCF_ID'] == TC], var)
        
        category = 'RI' if full_data[full_data['ATCF_ID'] == TC]['RI'].sum() > 0 else 'NI'
        
        results.append({
            'ATCF_ID': TC,
            'Variable': var,
            'Categoria': category,
            'Tendencia': stats_tc['Tendencia'],
            'Pendiente': stats_tc['Pendiente'],
            'Pendiente_s': stats_tc_s['Pendiente']
        })
df_stats_i = DataFrame(results)
df_slope_i = df_stats_i.drop(columns='Tendencia')
df_stats_i = df_stats_i.drop(columns=['Pendiente','Pendiente_s']).groupby(['Categoria','Variable','Tendencia']).count()
df_stats_i['total'] = df_stats_i.groupby(['Categoria','Variable']).transform('sum')
df_stats_i['percent'] = round(df_stats_i['ATCF_ID'] / df_stats_i['total'],4)*100
df_stats_i = df_stats_i.reset_index().pivot_table(
    index='Variable',
    columns=['Categoria', 'Tendencia'],
    values=['ATCF_ID', 'total', 'percent'],
    aggfunc='first'
)
df_stats_i = df_stats_i.reorder_levels([1, 2, 0], axis=1)
df_stats_i = df_stats_i.reindex(['RI', 'NI'], axis=1, level=0)
df_stats_i = df_stats_i.reindex(['decreasing', 'increasing', 'no trend'], axis=1, level=1)
df_stats_i.rename(columns={'ATCF_ID': 'count'}, level=2, inplace=True)
df_stats_i

#%% UNION DATOS DE PENDIENTE Y TASA
df_sloperate = df_slope_i.merge(df_rate,how='left')

#%% ANALISIS DE TENDENCIA CONJUNTA
results = []
for var in variables:
    ri_stats = stats_trend(data_ri, var)
    ni_stats = stats_trend(data_ni, var)

    count_samples_ri = df_stats_i['RI'][ri_stats['Tendencia']].loc[var]
    count_samples_ni = df_stats_i['NI'][ni_stats['Tendencia']].loc[var]
    
    # Combine the results for this variable
    combined_stats = concat({
        'RI': concat([ri_stats, count_samples_ri], axis=0),
        'NI': concat([ni_stats, count_samples_ni], axis=0)
    })
    
    results.append(combined_stats)

df_stats = concat(results, keys=variables, axis=1).T

#%% FIGURA 1 [ht_count,rb_count,ht_dist_mean,rb_dist_mean]
fig, axs = plt.subplots(2, 4, figsize=(18, 6.5), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.25})
axs = axs.flatten()

plot_configs = [
    # Primera fila (RI cases)
    (data_ri, 'ht_count', '#540b0e'),      
    (data_ri, 'rb_count', '#540b0e'),      
    (data_ri, 'ht_dist_mean', '#540b0e'),  
    (data_ri, 'rb_dist_mean', '#540b0e'),  
    
    # Segunda fila (NI cases)
    (data_ni, 'ht_count', '#4361ee'),      
    (data_ni, 'rb_count', '#4361ee'),      
    (data_ni, 'ht_dist_mean', '#4361ee'),  
    (data_ni, 'rb_dist_mean', '#4361ee'),  
]
for ax, (data, column, color) in zip(axs, plot_configs):
    for id in data['ATCF_ID'].unique():
        data_id = data.query(f'ATCF_ID == "{id}"')
        data_id.set_index('hours_from_max').sort_index()[column].plot(ax=ax,color=color,alpha=.25)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.axvline(0, color=(0.45, 0.45, 0.45), alpha=0.5, linestyle='dashed')
    ax.set_xlim(-100, 100)
    ax.set_xlabel('')
    ax.set_ylabel('')
for i, ax in enumerate(axs):
    var = ['ht_count', 'rb_count','ht_dist_mean', 'rb_dist_mean'][i % 4]
    is_ri = i < 4 
    row = df_stats.loc[var]
    add_stats_ts(ax, row,is_ri)
    add_pie_ts(ax, df_stats_i, var, is_ri)

# Add titles
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Mean Distance to Center of\nHot Towers', **title_style)
axs[3].set_title('Mean Distance to Center of\nConvective Systems', **title_style)

# Add y-axis labels
ylabel_style = dict(color=(0.45, 0.45, 0.45), fontsize=16, fontweight='bold')
axs[0].set_ylabel("RI Cases", **ylabel_style)
axs[4].set_ylabel("NI Cases", **ylabel_style)
plt.savefig(path_fig+f"Full_Convection_1.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Full_Convection_1.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)                
#%% FIGURA 2 [rb_area,rb_bt_std,ht_dist_std,rb_dist_std]
fig, axs = plt.subplots(2, 4, figsize=(18, 6.5), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.25})
axs = axs.flatten()

plot_configs = [
    # Primera fila (RI cases)
    (data_ri, 'rb_area', '#540b0e'),      
    (data_ri, 'rb_bt_std', '#540b0e'),          
    (data_ri, 'ht_dist_std', '#540b0e'),  
    (data_ri, 'rb_dist_std', '#540b0e'),  
    
    # Segunda fila (NI cases)
    (data_ni, 'rb_area', '#4361ee'),      
    (data_ni, 'rb_bt_std', '#4361ee'),      
    (data_ni, 'ht_dist_std', '#4361ee'),  
    (data_ni, 'rb_dist_std', '#4361ee'),  
]
for ax, (data, column, color) in zip(axs, plot_configs):
    for id in data['ATCF_ID'].unique():
        data_id = data.query(f'ATCF_ID == "{id}"')
        data_id.set_index('hours_from_max').sort_index()[column].plot(ax=ax,color=color,alpha=.25)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.axvline(0, color=(0.45, 0.45, 0.45), alpha=0.5, linestyle='dashed')
    ax.set_xlim(-100, 100)
    ax.set_xlabel('')
    ax.set_ylabel('')

for i, ax in enumerate(axs):
    var = ['rb_area','rb_bt_std','ht_dist_std','rb_dist_std'][i % 4]
    is_ri = i < 4 
    row = df_stats.loc[var]
    add_stats_ts(ax, row,is_ri)
    add_pie_ts(ax, df_stats_i, var, is_ri)

# Add titles
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
axs[0].set_title('Convective Systems Area\n', **title_style)
axs[1].set_title('Convective Systems BT Deviation\n', **title_style)
axs[2].set_title('Deviations Distance to Center of\nHot Towers', **title_style)
axs[3].set_title('Deviations Distance to Center of\nConvective Systems', **title_style)

# Add y-axis labels
ylabel_style = dict(color=(0.45, 0.45, 0.45), fontsize=16, fontweight='bold')
axs[0].set_ylabel("RI Cases", **ylabel_style)
axs[4].set_ylabel("NI Cases", **ylabel_style)
plt.savefig(path_fig+f"Full_Convection_2.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Full_Convection_2.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)       

# %% FIGURA 3 rate-slope
fig, axs = plt.subplots(2, 4, figsize=(18, 7), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
label_style = dict(color=(0.45, 0.45, 0.45), fontsize=13)
axs = axs.flatten()
for idx, (ax, var) in enumerate(zip(axs, variables)):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ratslo_al = df_sloperate[df_sloperate['Variable']==var]
    ratslo_ni = ratslo_al.query('Categoria=="NI"')
    ratslo_ri = ratslo_al.query('Categoria=="RI"')
    regplot(x=ratslo_al['rate'],y=ratslo_al['Pendiente'],
            ax=ax, ci=None, scatter=False,
            line_kws=dict(color=(0.45, 0.45, 0.45),alpha=.7))
    regplot(x=ratslo_ni['rate'],y=ratslo_ni['Pendiente'],x_jitter=.9,
            ax=ax, ci=80, scatter=True,
            scatter_kws=dict(color='#4361ee'),
            line_kws=dict(color='#4361ee',linestyle=':'))
    regplot(x=ratslo_ri['rate'],y=ratslo_ri['Pendiente'],x_jitter=.9,
            ax=ax, ci=80, scatter=True,
            scatter_kws=dict(color='#540b0e'),
            line_kws=dict(color='#540b0e',linestyle=':'))    
    ax.set_xlabel('')
    ax.set_ylabel('')
    if idx in [0,4]:
        ax.set_ylabel("Slope", **label_style)
    if idx in [4,5,6,7]:
        ax.set_xlabel("Intensification Rate [knots]", **label_style)
title_style = dict(color=(0.45, 0.45, 0.45), pad=5, fontsize=12, fontweight='bold')
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Mean Distance to Center of\nHot Towers', **title_style)
axs[3].set_title('Mean Distance to Center of\nConvective Systems', **title_style)
axs[4].set_title('Convective Systems Area\n', **title_style)
axs[5].set_title('Convective Systems BT Deviation\n', **title_style)
axs[6].set_title('Deviations Distance to Center of\nHot Towers', **title_style)
axs[7].set_title('Deviations Distance to Center of\nConvective Systems', **title_style)
plt.savefig(path_fig+f"Trend_rate_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Trend_rate_Convection.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)  

# %% FIGURA 4 vmax-slope
fig, axs = plt.subplots(2, 4, figsize=(18, 7), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
label_style = dict(color=(0.45, 0.45, 0.45), fontsize=13)
axs = axs.flatten()
for idx, (ax, var) in enumerate(zip(axs, variables)):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ratslo_al = df_sloperate[df_sloperate['Variable']==var]
    ratslo_ni = ratslo_al.query('Categoria=="NI"')
    ratslo_ri = ratslo_al.query('Categoria=="RI"')
    regplot(x=ratslo_al['VMAX'],y=ratslo_al['Pendiente'],
            ax=ax, ci=None, scatter=False,
            line_kws=dict(color=(0.45, 0.45, 0.45),alpha=.7))
    regplot(x=ratslo_ni['VMAX'],y=ratslo_ni['Pendiente'],x_jitter=.9,
            ax=ax, ci=80, scatter=True,
            scatter_kws=dict(color='#4361ee'),
            line_kws=dict(color='#4361ee',linestyle=':'))
    regplot(x=ratslo_ri['VMAX'],y=ratslo_ri['Pendiente'],x_jitter=.9,
            ax=ax, ci=80, scatter=True,
            scatter_kws=dict(color='#540b0e'),
            line_kws=dict(color='#540b0e',linestyle=':'))    
    ax.set_xlabel('')
    ax.set_ylabel('')
    if idx in [0,4]:
        ax.set_ylabel("Slope", **label_style)
    if idx in [4,5,6,7]:
        ax.set_xlabel("Max Intensity [knots]", **label_style)
title_style = dict(color=(0.45, 0.45, 0.45), pad=5, fontsize=12, fontweight='bold')
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Mean Distance to Center of\nHot Towers', **title_style)
axs[3].set_title('Mean Distance to Center of\nConvective Systems', **title_style)
axs[4].set_title('Convective Systems Area\n', **title_style)
axs[5].set_title('Convective Systems BT Deviation\n', **title_style)
axs[6].set_title('Deviations Distance to Center of\nHot Towers', **title_style)
axs[7].set_title('Deviations Distance to Center of\nConvective Systems', **title_style)
plt.savefig(path_fig+f"Trend_vmax_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Trend_vmax_Convection.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)  

# %% FIGURA 5 rb area #Ht
fig = plt.figure(figsize=(14.5,3),facecolor='white',edgecolor='white')
gs = GridSpec(1,2,wspace=.3,hspace=.4)
ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
ax3,ax4 = ax1.twinx(),ax2.twinx()
for idx,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='major', labelsize=9, 
                colors=(0.45, 0.45, 0.45))
    if idx in [0,2]:
        ax.spines['left'].set_color('#540b0e')
        ax.spines['right'].set_color('#ac171a')
    if idx in [1,3]:
        ax.spines['left'].set_color('#4361ee')
        ax.spines['right'].set_color('#93a5f6')
ax1.tick_params(axis='y', which='major', labelsize=9,colors='#540b0e')
ax2.tick_params(axis='y', which='major', labelsize=9,colors='#4361ee')
ax3.tick_params(axis='y', which='major', labelsize=9,colors='#ac171a')
ax4.tick_params(axis='y', which='major', labelsize=9,colors='#93a5f6')
ax1.plot(data_ri[data_ri['hours_from_max'].between(-100, 0)]['hours_from_max'],
         data_ri[data_ri['hours_from_max'].between(-100, 0)]['ht_count']['mean'], color='#540b0e', alpha=0.95)
ax2.plot(data_ni[data_ni['hours_from_max'].between(-100, 0)]['hours_from_max'],
         data_ni[data_ni['hours_from_max'].between(-100, 0)]['ht_count']['mean'], color='#4361ee', alpha=0.95)
ax3.plot(data_ri[data_ri['hours_from_max'].between(-100, 0)]['hours_from_max'],
         data_ri[data_ri['hours_from_max'].between(-100, 0)]['rb_area']['mean'],  color='#ac171a', alpha=0.95)
ax4.plot(data_ni[data_ni['hours_from_max'].between(-100, 0)]['hours_from_max'],
         data_ni[data_ni['hours_from_max'].between(-100, 0)]['rb_area']['mean'],  color='#93a5f6', alpha=0.95)

ax1.set_ylabel('Number of Hot Towers'   , color='#540b0e', fontsize=12)
ax2.set_ylabel('Number of Hot Towers'   , color='#4361ee', fontsize=12)
ax3.set_ylabel('Convective Systems Area', color='#ac171a', fontsize=12)
ax4.set_ylabel('Convective Systems Area', color='#93a5f6', fontsize=12)
ax1.set_title("RI Cases",color='#540b0e', pad=5, fontsize=12, fontweight='bold')
ax2.set_title("NI Cases",color='#4361ee', pad=5, fontsize=12, fontweight='bold')
plt.savefig(path_fig+f"Explore_cycles.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Explore_cycles.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)    

  

# %%
