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
from pandas import read_csv,DataFrame,concat,MultiIndex
from matplotlib import pyplot as plt
from fuction_tools import stats_trend,create_plots_ts,add_stats_ts,add_pie_ts
from fuction_tools import valores_comparacion
from seaborn import regplot,kdeplot,boxplot
from matplotlib.gridspec import GridSpec
from numpy import nan
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% PATHS
path_fig = '/var/data2/GOES_TRACK/FINAL/FIG/'
path_df = '/var/data2/GOES_TRACK/FINAL/DF_DATA/'
patha_data = '/home/dunievesr/Documents/UNAL/Final_codes/'
path_fig = '/home/dunievesr/Datos/TESIS/FIG/'
path_df = '/home/dunievesr/Datos/TESIS/DF_DATA/'
patha_data = '/home/dunievesr/Dropbox/UNAL/TESIS/Final_codes/'
letters = ['A', 'C', 'E', 'G', 'B', 'D', 'F', 'H']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i']

#%% FUNCIONES PEQUEÑAS
def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        if k == 'boxes':
            for box in v:
                box.set_facecolor('none')
                box.set_edgecolor(color_code)
                box.set_linewidth(0.8)
        else:
            plt.setp(plot_name.get(k), color=color_code, linewidth=0.8)

#%% DATOS CONVECTIVE
full_data = read_csv(path_df+'Convective_full_data.csv',index_col=[0])
full_data_scaled = read_csv(path_df+'Convective_full_data_scaled.csv',index_col=[0])
#full_data['hours_from_max'] = full_data['hours_from_max'].round(1)
summary_full_data = full_data_scaled.drop(columns='ATCF_ID').groupby(['RI','hours_from_max']).agg(['mean','std']).reset_index()
summary_full_data_ns = full_data.drop(columns='ATCF_ID').groupby(['RI','hours_from_max']).agg(['mean','std']).reset_index()
data_ni = summary_full_data[summary_full_data['RI']==0]
data_ri = summary_full_data[summary_full_data['RI']==1]
data_ni_ns = summary_full_data_ns[summary_full_data_ns['RI']==0]
data_ri_ns = summary_full_data_ns[summary_full_data_ns['RI']==1]

#%% REcorriendo cada ciclon
for tc in full_data['ATCF_ID'].unique():
    data_tc = full_data.loc[full_data['ATCF_ID']==tc]
    if data_tc['RI'].sum() !=0:
        fig = plt.figure(figsize=(10, 7), facecolor='none', edgecolor='none')
        ax = fig.add_subplot(111)
        data_tc.set_index('hours_from_max')[['ht_count','rb_count','rb_area','rb_bt_std']].plot(subplots=True,ax=ax)
        fig.suptitle(tc)
#%% DISTRIBUCIONES
fig, axs = plt.subplots(2, 4, figsize=(18, 6.5), facecolor='white', 
                        gridspec_kw={'wspace': 0.17,'hspace': 1.2})
axs = axs.flatten()

for idx_ax, (ax, column) in enumerate(zip(axs, ['ht_count', 'rb_count', 'rb_area', 'rb_bt_std',
                                                'ht_dist_mean', 'rb_dist_mean', 'ht_dist_std', 'rb_dist_std'])):
    data_copy = summary_full_data_ns[summary_full_data_ns['hours_from_max'].between(-100,0)].copy()
    data_copy.columns = MultiIndex.from_tuples([(c if isinstance(c, tuple) else (c, '')) for c in data_copy.columns])
    data_copy.loc[data_copy['hours_from_max'].between(-100,-75),'Range'] = '[100-75]'
    data_copy.loc[data_copy['hours_from_max'].between(-75,-50),'Range'] = '[75-50]'
    data_copy.loc[data_copy['hours_from_max'].between(-50,-25),'Range'] = '[50-25]'
    data_copy.loc[data_copy['hours_from_max'].between(-25,0),'Range'] = '[25,0]'
    data_copy = data_copy.loc[:, data_copy.columns.get_level_values(1).isin(['mean', ''])]
    data_copy = data_copy[['RI','Range',column]].droplevel(1,axis=1)
    kdeplot(data_ri_ns[data_ri_ns['hours_from_max'].between(-100,0)][column]['mean'],color='#540b0e',fill=True,ax=ax)
    kdeplot(data_ni_ns[data_ni_ns['hours_from_max'].between(-100,0)][column]['mean'],color='#4361ee',fill=True,ax=ax)
    ax.axvline(data_ri_ns[data_ri_ns['hours_from_max'].between(-100,0)][column]['mean'].mean(), color='#540b0e', alpha=0.5, linestyle=':')
    ax.axvline(data_ni_ns[data_ni_ns['hours_from_max'].between(-100,0)][column]['mean'].mean(), color='#4361ee', alpha=0.5, linestyle=':')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.set_ylabel('')
    # Insertar axes 1
    inset_ax1 = inset_axes(ax, width="100%", height="60%",
                        loc='upper right',
                        bbox_to_anchor=(0.025, .8, 1, 1),
                        bbox_transform=ax.transAxes)
    inset_ax1.patch.set_facecolor('none')

    if idx_ax in [0,4]:
        inset_ax1.set_ylabel('Hours Until\nMax Intensity ',color=(0.45, 0.45, 0.45),size=10)
    else:
        inset_ax1.set_ylabel('')
    inset_ax1.set_xlabel('')

    # Separar los datos por categoría de Range y por RI
    ranges = data_copy['Range'].unique()[::-1]
    data_by_range_ri0 = {}
    data_by_range_ri1 = {}

    for r in ranges:
        data_by_range_ri0[r] = data_copy[(data_copy['Range'] == r) & (data_copy['RI'] == 0)][column].values
        data_by_range_ri1[r] = data_copy[(data_copy['Range'] == r) & (data_copy['RI'] == 1)][column].values

    # Crear listas para los boxplots
    ri0_data = [data_by_range_ri0[r] for r in ranges]
    ri1_data = [data_by_range_ri1[r] for r in ranges]

    # Posiciones verticales con mayor separación (ajusta el 2.0 para cambiar el espaciado)
    positions_ri0 = array(range(len(ranges))) * 3.0 - 0.5
    positions_ri1 = array(range(len(ranges))) * 3.0 + 0.5

    # Crear los boxplots HORIZONTALES
    box_ri0 = inset_ax1.boxplot(ri0_data, 
                            positions=positions_ri0, 
                            widths=0.6, 
                            patch_artist=True,
                            flierprops={'markersize': 1},
                            vert=False)

    box_ri1 = inset_ax1.boxplot(ri1_data, 
                            positions=positions_ri1, 
                            widths=0.6, 
                            patch_artist=True,
                            flierprops={'markersize': 1},
                            vert=False) 

    inset_ax1.set_yticks(np.array(range(len(ranges))) * 3.0)
    inset_ax1.set_yticklabels(ranges)
    inset_ax1.set_ylim(-1, len(ranges) * 3.0)

    # Aplicar las propiedades a cada boxplot con nuestros colores específicos
    define_box_properties(box_ri0, '#4361ee')  # Color azul para RI=0
    define_box_properties(box_ri1, '#540b0e')  # Color rojo para RI=1

    for label in inset_ax1.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_x(-0.13)

    for spine in ['top', 'right']:
        inset_ax1.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        inset_ax1.spines[spine].set_color((0.45, 0.45, 0.45))
    
    inset_ax1.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    inset_ax1.set_xlim(ax.get_xlim())
    inset_ax1.set_xticklabels([])


axs[0].set_ylabel('Density',color=(0.45, 0.45, 0.45),size=10)
axs[4].set_ylabel('Density',color=(0.45, 0.45, 0.45),size=10)
axs[0].set_xlabel('Number of Hot Towers', color=(0.45, 0.45, 0.45),size=10)
axs[1].set_xlabel('Number of Convective Systems', color=(0.45, 0.45, 0.45),size=10)
axs[2].set_xlabel('Convective Systems Area', color=(0.45, 0.45, 0.45),size=10)
axs[3].set_xlabel('Convective Systems BT Deviation', color=(0.45, 0.45, 0.45),size=10)
axs[4].set_xlabel('Mean Distance to Center of\nHot Towers',color=(0.45, 0.45, 0.45),size=10)
axs[5].set_xlabel('Mean Distance to Center of\nConvective Systems',color=(0.45, 0.45, 0.45),size=10)
axs[6].set_xlabel('Deviations Distance to Center of\nHot Towers',color=(0.45, 0.45, 0.45),size=10)
axs[7].set_xlabel('Deviations Distance to Center of\nConvective Systems',color=(0.45, 0.45, 0.45),size=10)
plt.savefig(path_fig+f"distribution.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"distribution.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)

#%% DISTRIBUCIONES ESCALADAS
fig, axs = plt.subplots(2, 4, figsize=(18, 6.5), facecolor='white', 
                        gridspec_kw={'wspace': 0.17,'hspace': 1.2})
axs = axs.flatten()

for idx_ax, (ax, column) in enumerate(zip(axs, ['ht_count', 'rb_count', 'rb_area', 'rb_bt_std',
                                                'ht_dist_mean', 'rb_dist_mean', 'ht_dist_std', 'rb_dist_std'])):

    data_copy = summary_full_data[summary_full_data['hours_from_max'].between(-100,0)].copy()
    data_copy.columns = MultiIndex.from_tuples([(c if isinstance(c, tuple) else (c, '')) for c in data_copy.columns])
    data_copy.loc[data_copy['hours_from_max'].between(-100,-75),'Range'] = '[100-75]'
    data_copy.loc[data_copy['hours_from_max'].between(-75,-50),'Range'] = '[75-50]'
    data_copy.loc[data_copy['hours_from_max'].between(-50,-25),'Range'] = '[50-25]'
    data_copy.loc[data_copy['hours_from_max'].between(-25,0),'Range'] = '[25,0]'
    data_copy = data_copy.loc[:, data_copy.columns.get_level_values(1).isin(['mean', ''])]
    data_copy = data_copy[['RI','Range',column]].droplevel(1,axis=1)
    kdeplot(data_ri[data_ri['hours_from_max'].between(-100,0)][column]['mean'],color='#540b0e',fill=True,ax=ax)
    kdeplot(data_ni[data_ni['hours_from_max'].between(-100,0)][column]['mean'],color='#4361ee',fill=True,ax=ax)
    ax.axvline(data_ri[data_ri['hours_from_max'].between(-100,0)][column]['mean'].mean(), color='#540b0e', alpha=0.5, linestyle=':')
    ax.axvline(data_ni[data_ni['hours_from_max'].between(-100,0)][column]['mean'].mean(), color='#4361ee', alpha=0.5, linestyle=':')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.set_ylabel('')
    # Insertar axes 1
    inset_ax1 = inset_axes(ax, width="100%", height="60%",
                        loc='upper right',
                        bbox_to_anchor=(0.025, .8, 1, 1),
                        bbox_transform=ax.transAxes)
    inset_ax1.patch.set_facecolor('none')

    if idx_ax in [0,4]:
        inset_ax1.set_ylabel('Hours Until\nMax Intensity ',color=(0.45, 0.45, 0.45),size=10)
    else:
        inset_ax1.set_ylabel('')
    inset_ax1.set_xlabel('')

    # Separar los datos por categoría de Range y por RI
    ranges = data_copy['Range'].unique()[::-1]
    data_by_range_ri0 = {}
    data_by_range_ri1 = {}

    for r in ranges:
        data_by_range_ri0[r] = data_copy[(data_copy['Range'] == r) & (data_copy['RI'] == 0)][column].values
        data_by_range_ri1[r] = data_copy[(data_copy['Range'] == r) & (data_copy['RI'] == 1)][column].values

    # Crear listas para los boxplots
    ri0_data = [data_by_range_ri0[r] for r in ranges]
    ri1_data = [data_by_range_ri1[r] for r in ranges]

    # Posiciones verticales con mayor separación (ajusta el 2.0 para cambiar el espaciado)
    positions_ri0 = array(range(len(ranges))) * 3.0 - 0.5
    positions_ri1 = array(range(len(ranges))) * 3.0 + 0.5

    # Crear los boxplots HORIZONTALES
    box_ri0 = inset_ax1.boxplot(ri0_data, 
                            positions=positions_ri0, 
                            widths=0.6, 
                            patch_artist=True,
                            flierprops={'markersize': 1},
                            vert=False)

    box_ri1 = inset_ax1.boxplot(ri1_data, 
                            positions=positions_ri1, 
                            widths=0.6, 
                            patch_artist=True,
                            flierprops={'markersize': 1},
                            vert=False) 

    inset_ax1.set_yticks(np.array(range(len(ranges))) * 3.0)
    inset_ax1.set_yticklabels(ranges)
    inset_ax1.set_ylim(-1, len(ranges) * 3.0)

    # Aplicar las propiedades a cada boxplot con nuestros colores específicos
    define_box_properties(box_ri0, '#4361ee')  # Color azul para RI=0
    define_box_properties(box_ri1, '#540b0e')  # Color rojo para RI=1

    for label in inset_ax1.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_x(-0.13)

    for spine in ['top', 'right']:
        inset_ax1.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        inset_ax1.spines[spine].set_color((0.45, 0.45, 0.45))
    
    inset_ax1.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    inset_ax1.set_xlim(ax.get_xlim())
    inset_ax1.set_xticklabels([])

axs[0].set_ylabel('Density',color=(0.45, 0.45, 0.45),size=10)
axs[4].set_ylabel('Density',color=(0.45, 0.45, 0.45),size=10)
axs[0].set_xlabel('Number of Hot Towers', color=(0.45, 0.45, 0.45),size=10)
axs[1].set_xlabel('Number of Convective Systems', color=(0.45, 0.45, 0.45),size=10)
axs[2].set_xlabel('Convective Systems Area', color=(0.45, 0.45, 0.45),size=10)
axs[3].set_xlabel('Convective Systems BT Deviation', color=(0.45, 0.45, 0.45),size=10)
axs[4].set_xlabel('Mean Distance to Center of\nHot Towers',color=(0.45, 0.45, 0.45),size=10)
axs[5].set_xlabel('Mean Distance to Center of\nConvective Systems',color=(0.45, 0.45, 0.45),size=10)
axs[6].set_xlabel('Deviations Distance to Center of\nHot Towers',color=(0.45, 0.45, 0.45),size=10)
axs[7].set_xlabel('Deviations Distance to Center of\nConvective Systems',color=(0.45, 0.45, 0.45),size=10)
plt.savefig(path_fig+f"distribution_scale.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"distribution_scale.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)    


#%% DATOS TRAYECTORIES
rate_results = [] 
ships_select = read_csv(patha_data + 'trayectories12.csv',index_col=[0],parse_dates=[0])
for TC in full_data['ATCF_ID'].unique():
    rapid = ships_select[ships_select['ATCF_ID']==TC].sort_index()[['VMAX']].reset_index(drop=False)
    max_vmax = rapid['VMAX'].max()
    max_vmax_idx = rapid['VMAX'].idxmax()
    max_vmax_time = rapid.loc[max_vmax_idx, 'date_time']
    for obsi,val,lag in zip(valores_comparacion.keys(),valores_comparacion.values(),[2]):
        rapid[f'rate_{obsi}'] = rapid['VMAX'].sub(rapid['VMAX'].shift(lag))
        ri_interest = rapid.iloc[0:max_vmax_idx+1].query(f'rate_{obsi}>{val}')
        time_last = nan if ri_interest.empty else (max_vmax_time - ri_interest.iloc[-1]['date_time']).total_seconds() / 3600
        time_first = nan if ri_interest.empty else (max_vmax_time - ri_interest.iloc[0]['date_time']).total_seconds() / 3600
        rate_results.append({
            'ATCF_ID': TC,
            'rate': rapid[f'rate_{obsi}'].max(),
            'VMAX': max_vmax,
            'time_last':time_last,
            'time_first':time_first

            })
df_rate = DataFrame(rate_results)

#%%EXPLORACION DE TIEMPOS ANTES RAPIDA INTENSIFICACION
df_rate.dropna().select_dtypes('float').plot(kind='box')
df_rate.dropna().select_dtypes('float').describe()
df_rate.dropna().select_dtypes('float').mode()
df_rate.dropna().select_dtypes('float').mean()+12
times_rate = df_rate.dropna().select_dtypes('float')+12

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
    (data_ri, 'rb_area', '#540b0e'),
    (data_ri, 'rb_bt_std', '#540b0e'),    
             
    # Segunda fila (NI cases)
    (data_ni, 'ht_count', '#4361ee'),      
    (data_ni, 'rb_count', '#4361ee'),      
    (data_ni, 'rb_area', '#4361ee'),  
    (data_ni, 'rb_bt_std', '#4361ee'),  
]

for idx_ax, (ax, (data, column, color)) in enumerate(zip(axs, plot_configs)):
    create_plots_ts(ax, data, column, color)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.axvline(0, color=(0.45, 0.45, 0.45), alpha=0.5, linestyle='dashed')
    ax.text(0.0, 1.115, f'({letters[idx_ax]})', 
            transform=ax.transAxes,
            color = (0.45, 0.45, 0.45),
            fontsize=12, fontweight='bold', va='top', ha='left')
    if idx_ax in [0,1,2,3]:
        #for time in times_rate['time_first']:
        #    ax.axvline(-time, color='#ac171a', alpha=0.2, linestyle='solid')
        #for time in times_rate['time_last']:
        #    ax.axvline(-time, color='#5D3FD3', alpha=0.1, linestyle='solid')
        ax.axvline(-times_rate['time_first'].mean(), color='#ac171a', alpha=0.5, linestyle=':')
        ax.axvline(-times_rate['time_last'].mean(), color='#5D3FD3', alpha=0.5, linestyle=':')
        #ax.axvspan(-times_rate['time_first'].max(), -times_rate['time_first'].min(), alpha=0.1, color='#ac171a',zorder=0)
        #ax.axvspan(-times_rate['time_last'].max(), -times_rate['time_last'].min(), alpha=0.1, color='#5D3FD3',zorder=0)
        ax.axvspan(-(times_rate['time_first'].mean()+times_rate['time_first'].std()), -(times_rate['time_first'].mean()-times_rate['time_first'].std()), alpha=0.1, color='#ac171a',zorder=0)
        ax.axvspan(-(times_rate['time_last'].mean()+times_rate['time_last'].std()), -(times_rate['time_last'].mean()-times_rate['time_last'].std()), alpha=0.1, color='#5D3FD3',zorder=0)
        band_start = 12 * (-100 // 12)
        ticks_positions = list(range(int(band_start), 0, 12))
        ax.set_xticks(ticks_positions, minor=True)
        ax.tick_params(axis='x', which='minor', direction='in', length=8, width=1, color=(0.45, 0.45, 0.45,.2))
        ax.set_xticklabels([], minor=True)
        #for i, hour in enumerate(range(int(band_start), int(0), 12)):
        #    if i % 2 == 0:  # Sombrear solo bandas alternadas
        #        ax.axvspan(hour, hour + 12, alpha=0.2, color='lightgray',zorder=0)

    ax.set_xlim(-100, 100)
    ax.set_xlabel('')
    ax.set_ylabel('')

for i, ax in enumerate(axs):
    var = ['ht_count', 'rb_count','rb_area','rb_bt_std'][i % 4]
    is_ri = i < 4 
    row = df_stats.loc[var]
    add_stats_ts(ax, row,is_ri)
    add_pie_ts(ax, df_stats_i, var, is_ri)

# Add titles
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Convective Systems Area\n', **title_style)
axs[3].set_title('Convective Systems BT Deviation\n', **title_style)

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
         
    (data_ri, 'ht_dist_mean', '#540b0e'),  
    (data_ri, 'rb_dist_mean', '#540b0e'),  
    (data_ri, 'ht_dist_std', '#540b0e'),  
    (data_ri, 'rb_dist_std', '#540b0e'),  
    
    # Segunda fila (NI cases)
    (data_ni, 'ht_dist_mean', '#4361ee'),      
    (data_ni, 'rb_dist_mean', '#4361ee'),      
    (data_ni, 'ht_dist_std', '#4361ee'),  
    (data_ni, 'rb_dist_std', '#4361ee'),  
]
for idx_ax, (ax, (data, column, color)) in enumerate(zip(axs, plot_configs)):
    create_plots_ts(ax, data, column, color)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    ax.axvline(0, color=(0.45, 0.45, 0.45), alpha=0.5, linestyle='dashed')
    ax.text(0.0, 1.115, f'({letters[idx_ax]})', 
            transform=ax.transAxes,
            color = (0.45, 0.45, 0.45),
            fontsize=12, fontweight='bold', va='top', ha='left')
    
    if idx_ax in [0,1,2,3]:
        #for time in times_rate['time_first']:
        #    ax.axvline(-time, color='#ac171a', alpha=0.2, linestyle='solid')
        #for time in times_rate['time_last']:
        #    ax.axvline(-time, color='#5D3FD3', alpha=0.1, linestyle='solid')
        ax.axvline(-times_rate['time_first'].mean(), color='#ac171a', alpha=0.5, linestyle=':')
        ax.axvline(-times_rate['time_last'].mean(), color='#5D3FD3', alpha=0.5, linestyle=':')
        #ax.axvspan(-times_rate['time_first'].max(), -times_rate['time_first'].min(), alpha=0.1, color='#ac171a',zorder=0)
        #ax.axvspan(-times_rate['time_last'].max(), -times_rate['time_last'].min(), alpha=0.1, color='#5D3FD3',zorder=0)
        ax.axvspan(-(times_rate['time_first'].mean()+times_rate['time_first'].std()), -(times_rate['time_first'].mean()-times_rate['time_first'].std()), alpha=0.1, color='#ac171a',zorder=0)
        ax.axvspan(-(times_rate['time_last'].mean()+times_rate['time_last'].std()), -(times_rate['time_last'].mean()-times_rate['time_last'].std()), alpha=0.1, color='#5D3FD3',zorder=0)
        band_start = 12 * (-100 // 12)
        ticks_positions = list(range(int(band_start), 0, 12))
        ax.set_xticks(ticks_positions, minor=True)
        ax.tick_params(axis='x', which='minor', direction='in', length=8, width=1, color=(0.45, 0.45, 0.45,.2))
        ax.set_xticklabels([], minor=True)
        #for i, hour in enumerate(range(int(band_start), int(0), 12)):
        #    if i % 2 == 0:  # Sombrear solo bandas alternadas
        #        ax.axvspan(hour, hour + 12, alpha=0.2, color='lightgray',zorder=0)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('')
    ax.set_ylabel('')

for i, ax in enumerate(axs):
    var = ['ht_dist_mean', 'rb_dist_mean','ht_dist_std','rb_dist_std'][i % 4]
    is_ri = i < 4 
    row = df_stats.loc[var]
    add_stats_ts(ax, row,is_ri)
    add_pie_ts(ax, df_stats_i, var, is_ri)

# Add titles
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')

axs[0].set_title('Mean Distance to Center of\nHot Towers', **title_style)
axs[1].set_title('Mean Distance to Center of\nConvective Systems', **title_style)
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
    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
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
    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
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
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Asumiendo que tu dataframe se llama df y tiene la estructura mostrada
# Vamos a agrupar por variable para analizar cada una

def mean_test(df):
    resultados = {}
    
    # Obtener lista única de variables
    variables = df['Variable'].unique()
    
    for variable in variables:
        # Filtrar dataframe para esta variable
        df_var = df[df['Variable'] == variable]
        
        # Separar por categoría
        grupo_ni = df_var[df_var['Categoria'] == 'NI']['Pendiente']
        grupo_ri = df_var[df_var['Categoria'] == 'RI']['Pendiente']
        
        # Prueba t para comparar pendientes entre grupos
        _, p_valor = stats.ttest_ind(grupo_ni, grupo_ri, equal_var=False)
        
        # También podemos hacer una prueba no paramétrica (Mann-Whitney)
        _, p_valor_mw = stats.mannwhitneyu(grupo_ni, grupo_ri)
        
        resultados[variable] = {
            'media_NI': grupo_ni.mean(),
            'media_RI': grupo_ri.mean(),
            'sentido_NI': "positiva" if grupo_ni.mean() > 0 else "negativa",
            'sentido_RI': "positiva" if grupo_ri.mean() > 0 else "negativa",
            'comportamiento_opuesto': grupo_ni.mean() * grupo_ri.mean() < 0,
            'p_valor_t': p_valor,
            'p_valor_mw': p_valor_mw,
            'diferencia_significativa': p_valor < 0.05 or p_valor_mw < 0.05
        }
        
    return resultados
# %%
da = mean_test(df_slope_i)
DataFrame(da)[variables[0:4]]

# %%
def bootstrap_pendientes_por_variable(df, n_bootstrap=1000):
    resultados = {}
    variables = df['Variable'].unique()
    
    for variable in variables:
        # Filtrar dataframe para esta variable
        df_var = df[df['Variable'] == variable]
        
        # Obtener pendientes por grupo
        pendientes_ni = df_var[df_var['Categoria'] == 'NI']['Pendiente'].values
        pendientes_ri = df_var[df_var['Categoria'] == 'RI']['Pendiente'].values
        
        # Calcular diferencia observada
        diff_observada = np.mean(pendientes_ri) - np.mean(pendientes_ni)
        
        # Para almacenar resultados de bootstrap
        bootstrap_ni = []
        bootstrap_ri = []
        bootstrap_diff = []
        sentidos_opuestos = 0
        
        # Iteraciones de bootstrap
        for _ in range(n_bootstrap):
            # Muestreo con reemplazo
            muestra_ni = np.random.choice(pendientes_ni, size=len(pendientes_ni), replace=True)
            muestra_ri = np.random.choice(pendientes_ri, size=len(pendientes_ri), replace=True)
            
            # Calcular media de cada muestra
            media_ni = np.mean(muestra_ni)
            media_ri = np.mean(muestra_ri)
            
            # Guardar resultados
            bootstrap_ni.append(media_ni)
            bootstrap_ri.append(media_ri)
            bootstrap_diff.append(media_ri - media_ni)
            
            # Verificar si tienen sentidos opuestos
            if media_ni * media_ri < 0:
                sentidos_opuestos += 1
        
        # Calcular intervalos de confianza (95%)
        ci_ni = np.percentile(bootstrap_ni, [2.5, 97.5])
        ci_ri = np.percentile(bootstrap_ri, [2.5, 97.5])
        ci_diff = np.percentile(bootstrap_diff, [2.5, 97.5])
        
        # Probabilidad de sentidos opuestos
        prob_opuestos = sentidos_opuestos / n_bootstrap
        
        # Diferencia significativa basada en si el CI incluye 0
        diferencia_significativa = (ci_diff[0] > 0) or (ci_diff[1] < 0)
        
        # Calcular p-valor bootstrap correctamente
        # Proporción de valores bootstrap más extremos que 0 (en la dirección de la diferencia observada)
        if diff_observada > 0:
            p_valor = np.mean([d <= 0 for d in bootstrap_diff])
        else:
            p_valor = np.mean([d >= 0 for d in bootstrap_diff])
        
        resultados[variable] = {
            'media_NI': np.mean(pendientes_ni),
            'media_RI': np.mean(pendientes_ri),
            'ci_NI': ci_ni,
            'ci_RI': ci_ri,
            'ci_diferencia': ci_diff,
            'media_diferencia_bootstrap': np.mean(bootstrap_diff), 
            'prob_sentidos_opuestos': prob_opuestos,
            'diferencia_significativa': diferencia_significativa,
            'p_valor_bootstrap': p_valor
        }
    
    return resultados
# %%
ese = bootstrap_pendientes_por_variable(df_slope_i)
DataFrame(ese)[['ht_count', 'rb_count','rb_area','rb_bt_std']]
# %%


#%%
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def bootstrap_regresion_pendientes_vs_intensidad(df, var_x, categoria_col, n_bootstrap=1000):
    """
    Analiza la diferencia entre las pendientes de regresión (pendiente vs intensidad) entre grupos.
    
    Parámetros:
    df: DataFrame con los datos
    var_y: Nombre de la variable independiente ('rate' o 'VMAX')
    categoria_col: Nombre de la columna que contiene la categoría (NI/RI)
    n_bootstrap: Número de iteraciones bootstrap
    """
    resultados = {}
    
    # Obtener lista única de variables
    variables = df['Variable'].unique()
    
    for variable in variables:
        # Filtrar dataframe para esta variable
        df_var = df[df['Variable'] == variable]
        
        # Separar por categoría
        df_ni = df_var[df_var[categoria_col] == 'NI']
        df_ri = df_var[df_var[categoria_col] == 'RI']
        
        # Verificar si hay suficientes datos
        if len(df_ni) < 3 or len(df_ri) < 3:
            continue
        
        # Calcular pendientes originales
        slope_ni, intercept_ni, r_ni, p_ni, se_ni = stats.linregress(df_ni['Pendiente'],df_ni[var_x])
        slope_ri, intercept_ri, r_ri, p_ri, se_ri = stats.linregress(df_ri['Pendiente'],df_ri[var_x])
        
        # Diferencia original
        diff_original = slope_ri - slope_ni
        
        # Almacenar resultados bootstrap
        bootstrap_ni = []
        bootstrap_ri = []
        bootstrap_diff = []
        sentidos_opuestos = 0
        
        # Ejecutar bootstrap
        for _ in range(n_bootstrap):
            # Muestreo con reemplazo
            muestra_ni = df_ni.sample(len(df_ni), replace=True)
            muestra_ri = df_ri.sample(len(df_ri), replace=True)
            
            # Calcular pendientes en cada muestra
            try:
                b_slope_ni, _, _, _, _ = stats.linregress(muestra_ni['Pendiente'],muestra_ni[var_x])
                b_slope_ri, _, _, _, _ = stats.linregress(muestra_ri['Pendiente'],muestra_ri[var_x])
                
                # Guardar resultados
                bootstrap_ni.append(b_slope_ni)
                bootstrap_ri.append(b_slope_ri)
                bootstrap_diff.append(b_slope_ri - b_slope_ni)
                
                # Verificar si tienen sentidos opuestos
                if b_slope_ni * b_slope_ri < 0:
                    sentidos_opuestos += 1
            except:
                # En caso de error en el cálculo de regresión (puede ocurrir con muestras problemáticas)
                continue
        
        # Calcular intervalos de confianza (95%)
        ci_ni = np.percentile(bootstrap_ni, [2.5, 97.5])
        ci_ri = np.percentile(bootstrap_ri, [2.5, 97.5])
        ci_diff = np.percentile(bootstrap_diff, [2.5, 97.5])
        
        # Probabilidad de sentidos opuestos
        prob_opuestos = sentidos_opuestos / len(bootstrap_diff) if len(bootstrap_diff) > 0 else 0
        
        # Calcular p-valor
        if diff_original > 0:
            p_valor = np.mean([d <= 0 for d in bootstrap_diff]) if len(bootstrap_diff) > 0 else 1.0
        else:
            p_valor = np.mean([d >= 0 for d in bootstrap_diff]) if len(bootstrap_diff) > 0 else 1.0
        
        # Determinar si la diferencia es significativa
        diferencia_significativa = (0 < ci_diff[0]) or (0 > ci_diff[1])
        
        resultados[variable] = {
            'var_x': var_x,
            'media_NI': df_ni['Pendiente'].mean(),
            'media_RI': df_ri['Pendiente'].mean(),
            'pendiente_NI': slope_ni,
            'pendiente_RI': slope_ri,
            'r_value_NI': r_ni,
            'r_value_RI': r_ri,
            'p_value_NI': p_ni,
            'p_value_RI': p_ri,
            'ci_NI': ci_ni,
            'ci_RI': ci_ri,
            'ci_diferencia': ci_diff,
            'prob_sentidos_opuestos': prob_opuestos,
            'diferencia_significativa': diferencia_significativa,
            'p_valor': p_valor
        }
    
    return resultados

# Para analizar tanto para rate como para VMAX
def analizar_ambas_variables(df, n_bootstrap=1000):
    """
    Realiza análisis para ambas variables independientes: rate y VMAX
    """
    resultados_rate = bootstrap_regresion_pendientes_vs_intensidad(df, 'rate', 'Categoria', n_bootstrap)
    resultados_vmax = bootstrap_regresion_pendientes_vs_intensidad(df, 'VMAX', 'Categoria', n_bootstrap)
    
    return {
        'rate': resultados_rate,
        'VMAX': resultados_vmax
    }

# Para convertir los resultados en DataFrame
def convertir_resultados_a_dataframe(resultados_combinados):
    filas = []
    
    for var_x_nombre, resultados in resultados_combinados.items():
        for variable, res in resultados.items():
            filas.append({
                'Variable_Y': variable,
                'Variable_X': var_x_nombre,
                'Pendiente_NI': res['pendiente_NI'],
                'Pendiente_RI': res['pendiente_RI'],
                'R_NI': res['r_value_NI'],
                'R_RI': res['r_value_RI'],
                'P_Reg_NI': res['p_value_NI'],
                'P_Reg_RI': res['p_value_RI'],
                'CI_NI_Lower': res['ci_NI'][0],
                'CI_NI_Upper': res['ci_NI'][1],
                'CI_RI_Lower': res['ci_RI'][0],
                'CI_RI_Upper': res['ci_RI'][1],
                'CI_Diff_Lower': res['ci_diferencia'][0],
                'CI_Diff_Upper': res['ci_diferencia'][1],
                'P_Valor_Bootstrap': res['p_valor'],
                'Prob_Sentidos_Opuestos': res['prob_sentidos_opuestos'],
                'Diferencia_Significativa': res['diferencia_significativa']
            })
    
    return pd.DataFrame(filas)

# Función para generar tabla LaTeX
def generar_tabla_latex(df_resultados, var_x):
    df_filtrado = df_resultados[df_resultados['Variable_X'] == var_x]
    
    tabla_latex = "\\begin{table}[h]\n"
    tabla_latex += "\\centering\n"
    tabla_latex += f"\\caption{{Comparative analysis of regression slopes between RI and NI cyclones for {var_x} vs. various metrics. Testing the null hypothesis $H_0$: No significant difference exists between regression slopes of RI and NI cyclones. Analysis performed using bootstrap resampling ($n=1000$).}}\n"
    tabla_latex += "\\footnotesize\n"
    tabla_latex += "\\begin{tabular}{lcccccc}\n"
    tabla_latex += "\\hline\n"
    tabla_latex += "\\textbf{Variable} & \\multicolumn{2}{c}{\\textbf{Regression Slope}} & \\textbf{95\\% CI of} & \\textbf{p-value} & \\textbf{Prob. opposite} & \\textbf{Significant} \\\\\n"
    tabla_latex += " & \\textbf{NI (95\\% CI)} & \\textbf{RI (95\\% CI)} & \\textbf{difference} & & \\textbf{directions} & \\textbf{difference} \\\\\n"
    tabla_latex += "\\hline\n"
    
    for _, row in df_filtrado.iterrows():
        var = row['Variable_Y']
        ni_slope = f"{row['Pendiente_NI']:.4f}"
        ri_slope = f"{row['Pendiente_RI']:.4f}"
        ni_ci = f"[{row['CI_NI_Lower']:.4f}, {row['CI_NI_Upper']:.4f}]"
        ri_ci = f"[{row['CI_RI_Lower']:.4f}, {row['CI_RI_Upper']:.4f}]"
        diff_ci = f"[{row['CI_Diff_Lower']:.4f}, {row['CI_Diff_Upper']:.4f}]"
        p_valor = f"{row['P_Valor_Bootstrap']:.4f}"
        prob_opuestos = f"{row['Prob_Sentidos_Opuestos']:.3f}"
        significativo = "Yes" if row['Diferencia_Significativa'] else "No"
        
        tabla_latex += f"{var} & {ni_slope} & {ri_slope} & {diff_ci} & {p_valor} & {prob_opuestos} & {significativo} \\\\\n"
        tabla_latex += f" & {ni_ci} & {ri_ci} & & & & \\\\\n"
    
    tabla_latex += "\\hline\n"
    tabla_latex += "\\end{tabular}\n"
    tabla_latex += f"\\label{{tab:slope_comparison_{var_x.lower()}}}\n"
    tabla_latex += "\\end{table}"
    
    return tabla_latex

# Uso de las funciones
resultados_combinados = analizar_ambas_variables(df_sloperate)
df_resultados = convertir_resultados_a_dataframe(resultados_combinados)
tabla_rate = generar_tabla_latex(df_resultados, 'rate')
tabla_vmax = generar_tabla_latex(df_resultados, 'VMAX')
print(tabla_rate)
print("\n\n")
print(tabla_vmax)

# %%
import matplotlib.pyplot as plt

# Asumiendo que ya tienes la figura creada con los gráficos
# Para cada subplot, agrega un cuadro de texto con la información estadística

def agregar_estadisticas(ax, resultados_var):
    """
    Agrega un cuadro de texto con estadísticas en un subplot
    
    ax: el objeto Axes donde agregar el texto
    resultados_var: diccionario con resultados bootstrap para esta variable
    panel_letra: etiqueta del panel (A, B, C, etc.)
    """
    # Determinar asteriscos para significancia
    if resultados_var['p_valor'] < 0.001:
        sig_stars = '***'
    elif resultados_var['p_valor'] < 0.01:
        sig_stars = '**'
    elif resultados_var['p_valor'] < 0.05:
        sig_stars = '*'
    else:
        sig_stars = 'ns'
    
    # Formato del texto
    #texto = f"{panel_letra}\n"
    texto = f"Diff: {resultados_var['pendiente_RI'] - resultados_var['pendiente_NI']:.3f} {sig_stars}\n"
    texto += f"P-value: {resultados_var['p_valor']:.3f}\n"
    texto += f"Mean RI: {resultados_var.get('media_RI'):.2e}\n"
    texto += f"Mean NI: {resultados_var.get('media_NI'):.2e}"
    # Posición del texto (esquina superior izquierda)
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.6)
    ax.text(0.03, 0.99, texto, transform=ax.transAxes, fontsize=9,
            color=(0.45, 0.45, 0.45),
            verticalalignment='top',bbox=props)
# %%
#%% FUCION GRAFICOS Y SLOPE
fig, axs = plt.subplots(2, 4, figsize=(18, 7), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
label_style = dict(color=(0.45, 0.45, 0.45), fontsize=13)
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
axs = axs.flatten()
for idx, (ax, var) in enumerate(zip(axs, ['ht_count', 'rb_count','rb_area','rb_bt_std']*2)):
    ax.grid(color='gray', linestyle=':', linewidth=0.4, alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    if idx in [0,1,2,3]:
        agregar_estadisticas(ax, resultados_combinados['VMAX'][var])
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

        ax.set_xlabel("Max Intensity [knots]", **label_style)
    if idx in [4,5,6,7]:
        agregar_estadisticas(ax, resultados_combinados['rate'][var])
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
        ax.set_xlabel("Intensification Rate [knots]", **label_style)
    #ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.0, 1.115, f'({letters[idx]})', 
            transform=ax.transAxes,
            color = (0.45, 0.45, 0.45),
            fontsize=12, fontweight='bold', va='top', ha='left')
    if idx in [0,4]:
        ax.set_ylabel("Slope", **label_style)
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Convective Systems Area\n', **title_style)
axs[3].set_title('Convective Systems BT Deviation\n', **title_style)
plt.savefig(path_fig+f"Trend_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Trend_Convection.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)  

#%% FUCION GRAFICOS X SLOPE
fig, axs = plt.subplots(2, 4, figsize=(18, 7), facecolor='white', 
                        gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
label_style = dict(color=(0.45, 0.45, 0.45), fontsize=13)
title_style = dict(color=(0.45, 0.45, 0.45), pad=8, fontsize=12, fontweight='bold')
axs = axs.flatten()
for idx, (ax, var) in enumerate(zip(axs, ['ht_count', 'rb_count','rb_area','rb_bt_std']*2)):
    ax.grid(color='gray', linestyle=':', linewidth=0.4, alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((0.45, 0.45, 0.45))
    
    ax.tick_params(axis='both', which='major', labelsize=7.5, 
                  colors=(0.45, 0.45, 0.45))
    if idx in [0,1,2,3]:
        agregar_estadisticas(ax, resultados_combinados['VMAX'][var])
        ratslo_al = df_sloperate[df_sloperate['Variable']==var]
        ratslo_ni = ratslo_al.query('Categoria=="NI"')
        ratslo_ri = ratslo_al.query('Categoria=="RI"')
        regplot(y=ratslo_al['VMAX'],x=ratslo_al['Pendiente'],
                ax=ax, ci=None, scatter=False,
                line_kws=dict(color=(0.45, 0.45, 0.45),alpha=.7))
        regplot(y=ratslo_ni['VMAX'],x=ratslo_ni['Pendiente'],y_jitter=.9,
                ax=ax, ci=80, scatter=True,
                scatter_kws=dict(color='#4361ee'),
                line_kws=dict(color='#4361ee',linestyle=':'))
        regplot(y=ratslo_ri['VMAX'],x=ratslo_ri['Pendiente'],y_jitter=.9,
                ax=ax, ci=80, scatter=True,
                scatter_kws=dict(color='#540b0e'),
                line_kws=dict(color='#540b0e',linestyle=':'))  

        ax.set_xlabel("Slope [#/time]", **label_style)
    if idx in [4,5,6,7]:
        agregar_estadisticas(ax, resultados_combinados['rate'][var])
        ratslo_al = df_sloperate[df_sloperate['Variable']==var]
        ratslo_ni = ratslo_al.query('Categoria=="NI"')
        ratslo_ri = ratslo_al.query('Categoria=="RI"')
        regplot(y=ratslo_al['rate'],x=ratslo_al['Pendiente'],
                ax=ax, ci=None, scatter=False,
                line_kws=dict(color=(0.45, 0.45, 0.45),alpha=.7))
        regplot(y=ratslo_ni['rate'],x=ratslo_ni['Pendiente'],y_jitter=.9,
                ax=ax, ci=80, scatter=True,
                scatter_kws=dict(color='#4361ee'),
                line_kws=dict(color='#4361ee',linestyle=':'))
        regplot(y=ratslo_ri['rate'],x=ratslo_ri['Pendiente'],y_jitter=.9,
                ax=ax, ci=80, scatter=True,
                scatter_kws=dict(color='#540b0e'),
                line_kws=dict(color='#540b0e',linestyle=':'))    
        ax.set_xlabel("Slope [#/time]", **label_style)
    #ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.0, 1.115, f'({letters[idx]})', 
            transform=ax.transAxes,
            color = (0.45, 0.45, 0.45),
            fontsize=12, fontweight='bold', va='top', ha='left')
axs[0].set_ylabel("Max Intensity [knots]", **label_style)
axs[4].set_ylabel("Intensification Rate [knots]", **label_style)
axs[0].set_title('Number of Hot Towers\n', **title_style)
axs[1].set_title('Number of Convective Systems\n', **title_style)
axs[2].set_title('Convective Systems Area\n', **title_style)
axs[3].set_title('Convective Systems BT Deviation\n', **title_style)
plt.savefig(path_fig+f"Trend_Convection.png",pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_fig+f"Trend_Convection.pdf",pad_inches=0.1,bbox_inches='tight',dpi=250)  
# %%
