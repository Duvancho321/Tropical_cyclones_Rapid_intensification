"""
Codigo para hacer graficos basicos

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.1"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__refereces__:
__changues__:
    - [2024-12-14][Duvan]: Primera version del codigo.
"""
#%% MODULOS 
from pandas import read_csv
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from cartopy import crs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from fuction_tools import centrar_longitud,asignar_colores
from fuction_tools import names,paleta_amarillo_verde_azul,paleta_cian_rojo
from matplotlib.ticker import FixedLocator,AutoLocator

#%% PATHS
patha_data = '/home/dunievesr/Dropbox/UNAL/TESIS/Final_codes/'
path_fig = '/home/dunievesr/Datos/TESIS/FIG/'

# %% DEFINICION  DE VARIABLE Y UNIDADES
variables = {'SHRG':'[knots]','IKE':'','USA_WIND':'[knots]','NTMX':'[°C]','DIVC':'[sec-1 * 10**7]','VVAC':'[m/s]','REFC':'[m/sec/day]',
             'VVAV':'[m/s]','DSTA':'[°C]','USA_RMW':'[nmile]','USA_PRES':'[hPa]','SHRS':'[knots]','CSST':'[°C]',
             'RHLO':'[%]','G150':'[°C]','PDI':'','PENV':'[hPa]','DTL':'[Km]','RSST':'[°C]','RHMD':'[%]',
             'DIST2LAND':'[m]','VMPI':'[knots]','NOHC':'[kJ/cm²]','LANDFALL':'[Km]','ACE':'', 'MTPW':'[mm]',
             'VMAX':'[knots]','PSLV':'[hPa]','VMFX':'[m/s]','DSST':'[°C]','RHHI':'[%]','USA_ROCI':'[m]','SHRD':'[knots]',
             'STORM_SPEED':'[m/s]','CFLX':'[Dry air predictor]','NDML':'[m]','NDTX':'[m]'}
cuencas = ['AL','EP','WP']

#%% Fondo de mapa (all states)
deptos = cfeature.ShapelyFeature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes',
   '10m', edgecolor='gray', facecolor='none').geometries(),
    crs.PlateCarree(), facecolor='none')


#%% DATOS
ships_select = read_csv(patha_data + 'trayectories12.csv',index_col=[0],parse_dates=[0]).sort_index()
ships_select['BASIN'] = ships_select['ATCF_ID'].str[0:2]
ships_select['paso'] = (ships_select.groupby('ATCF_ID').cumcount() * 6)
ships_select['max_intensity_date'] = ships_select.groupby('ATCF_ID')['VMAX'].transform(
    lambda x: x.index[x == x.max()][0])
ships_select['hours_from_max']  = (ships_select.index - ships_select['max_intensity_date']).dt.total_seconds() / 3600
ships_select.loc[ships_select['BASIN'] == 'EP', 'TLON'] = -ships_select.loc[ships_select['BASIN'] == 'EP', 'TLON']
ships_select.loc[ships_select['BASIN'] == 'AL', 'TLON'] = -ships_select.loc[ships_select['BASIN'] == 'AL', 'TLON']

#%% EXTEND
limits = ships_select.groupby('BASIN').agg({'TLON': ['min', 'max']})['TLON']
AL_L = limits.loc['AL'].values
EP_L = limits.loc['EP'].values
WP_L = limits.loc['WP'].values

#%% Asignación de colores
colores = ships_select.groupby(['ATCF_ID','FINAL_O_INTENSITY'])[['VMAX']].max().reset_index().sort_values(['FINAL_O_INTENSITY','VMAX']).reset_index(drop=True)
colores.loc[colores['FINAL_O_INTENSITY']=='RI', "color"] = asignar_colores(colores[colores['FINAL_O_INTENSITY']=='RI'].index, paleta_cian_rojo) 
colores.loc[colores['FINAL_O_INTENSITY']=='NI', "color"] = asignar_colores(colores[colores['FINAL_O_INTENSITY']=='NI'].index, paleta_amarillo_verde_azul) 
colores.set_index('ATCF_ID',inplace=True)

#%% AJUSTES
max_step = ships_select['paso'].max()

#%% MAPA DE TRAYECTORIAS
fig = plt.figure(figsize=(16,3),facecolor='none',edgecolor='none')
gs = GridSpec(1, 7, width_ratios=[1, 1, 1, 0.1, 1, 1,1],wspace=0.06)
ax1,ax2,ax3   = (fig.add_subplot(gs[0,0],projection=crs.PlateCarree(central_longitude=180)),
                 fig.add_subplot(gs[0,1],projection=crs.PlateCarree(central_longitude=180)),
                 fig.add_subplot(gs[0,2],projection=crs.PlateCarree(central_longitude=180)))

ax4,ax5,ax6   = (fig.add_subplot(gs[0,4],projection=crs.PlateCarree(central_longitude=180)),
                 fig.add_subplot(gs[0,5],projection=crs.PlateCarree(central_longitude=180)),
                 fig.add_subplot(gs[0,6],projection=crs.PlateCarree(central_longitude=180)))
kwargs_lines = dict(marker=[(-1, -.9), (1, .9)], markersize=10,
                    linestyle="none", color=(0.45, 0.45, 0.45), mec=(0.45, 0.45, 0.45), mew=1, clip_on=False)

for i,ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax.set_aspect('auto')
    gl = ax.gridlines(crs=crs.PlateCarree(),draw_labels=True,linestyle=':')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': (0.45, 0.45, 0.45)}
    gl.ylabel_style = {'size': 8, 'color': (0.45, 0.45, 0.45)}
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = True if i == 5 else False 
    gl.left_labels = True if i == 0 else False
    gl.bottom_labels = True
    ax.coastlines(color='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('none')
    ax.set_ylim(-10,60)
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray',alpha=.6)
    if i in [0,1,3,4]:
        ax.plot([1], [0], transform=ax.transAxes, **kwargs_lines)
        ax.plot([1], [0.5], transform=ax.transAxes, **kwargs_lines)
        ax.plot([1], [1], transform=ax.transAxes, **kwargs_lines)
    if i in [1,2,4,5]:
        ax.plot([0], [0], transform=ax.transAxes, **kwargs_lines) 
        ax.plot([0], [0.5], transform=ax.transAxes, **kwargs_lines) 
        ax.plot([0], [1], transform=ax.transAxes, **kwargs_lines) 
    if i in [0,3]:
        ax.set_xlim(centrar_longitud(WP_L))
    if i in [1,4]:
        ax.set_xlim(centrar_longitud(EP_L-.7))
    if i in [2,5]:
        ax.set_xlim(centrar_longitud(AL_L))

    ax.add_feature(deptos,linewidth=0.1, edgecolor='black',linestyle='dotted')
RI_lines,RI_names = [], []
NI_lines,NI_names = [], []

for TC in ships_select['ATCF_ID'].unique():
    data = ships_select[ships_select['ATCF_ID']==TC][['paso','ATCF_ID','TLON','TLAT','VMAX','VMPI','SHRD','RSST','FINAL_O_INTENSITY','MNCDPC','BASIN']]
    intensity = data['FINAL_O_INTENSITY'].iloc[0]
    case = data['MNCDPC'].iloc[0]>=4
    basin = data['BASIN'].iloc[0]
    linestyle = '-' if case else ':'
    marker = 'o' if case else 'v'
    lon =  data['TLON']
    lat =  data['TLAT']
    if intensity == 'RI':
        if basin == 'WP':
            ax1.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        if basin == 'EP':
            ax2.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        if basin == 'AL':
            ax3.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        line = Line2D([0], [0], color=colores.loc[TC]['color'], lw=1.3, linestyle=linestyle)
        RI_lines.append(line);RI_names.append(f"{names[TC]} {TC[-4:]}")
    if intensity == 'NI':
        if basin == 'WP':
            ax4.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        if basin == 'EP':
            ax5.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        if basin == 'AL':
            ax6.plot(centrar_longitud(lon),lat,color=colores.loc[TC]['color'],alpha=.5,linewidth=1.2,marker=marker,markersize=2.2)
        line = Line2D([0], [0], color=colores.loc[TC]['color'], lw=1.3, linestyle=linestyle)
        NI_lines.append(line);NI_names.append(f"{names[TC]} {TC[-4:]}")
#Titulos
ax2.set_title(f'Rapid Intensification Cases' , color="#4361ee", pad=8, fontsize=13,fontweight='bold')
ax5.set_title(f'Non Rapid Intensification Cases', color="#e09f3e", pad=8, fontsize=13,fontweight='bold')
#%Leyendas
line1 = Line2D([0], [0], color=(0.45, 0.45, 0.45), lw=1.5, linestyle='-',marker='o')
line2 = Line2D([0], [0], color=(0.45, 0.45, 0.45), lw=1.5, linestyle=':',marker='v')
RI_LEG = fig.legend(RI_lines, RI_names,
                   bbox_to_anchor=(0.33, 1.45),frameon=False,
                   handlelength=2,loc='upper center', ncol=3)
NI_LEG = fig.legend(NI_lines, NI_names,
                   bbox_to_anchor=(0.715, 1.45),frameon=False,
                   handlelength=2,loc='upper center', ncol=3)
for leg in [RI_LEG,NI_LEG]:
    for text in leg.get_texts():
        text.set_color((0.45, 0.45, 0.45))
        text.set_fontsize(8)
plt.savefig(path_fig + f'SHIPS_FINAL12.pdf',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)
plt.savefig(path_fig + f'SHIPS_FINAL12.png',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)

# %% CONDICIONES 1
fig = plt.figure(figsize=(16,6),facecolor='none',edgecolor='none')
gs = GridSpec(2,2,wspace=.03,hspace=.06)#,height_ratios=[1,1,1,0.05])#,width_ratios=[1,0.05])#,height_ratios=[1,1,0.05])
ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])
ax3,ax4 = fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])
for i,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.spines['top'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['right'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.set_xlim(-100,100)
    ax.grid(True,linestyle=':', color='gray', alpha=0.2)
    if i in [0,2]:
        ax.set_xticklabels([])  
        ax.set_xlabel('')
    if i in [2,3]:
        ax.set_yticklabels([])  
        ax.set_ylabel('')
    ax.axvline(0,color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
for TC in ships_select['ATCF_ID'].unique():
    data = ships_select[ships_select['ATCF_ID']==TC][['paso','ATCF_ID','TLON','TLAT','VMAX','VMPI','SHRD','RSST','FINAL_O_INTENSITY','MNCDPC','BASIN','hours_from_max']]
    intensity = data['FINAL_O_INTENSITY'].iloc[0]
    case = data['MNCDPC'].iloc[0]>=4
    basin = data['BASIN'].iloc[0]
    linestyle = '-' if case else ':'
    marker = 'o' if case else 'v'
    if intensity == 'RI':
        ax1.plot(data['hours_from_max'] ,data['RSST'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax2.plot(data['hours_from_max'] ,data['SHRD'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=4)
    if intensity == 'NI':
        ax3.plot(data['hours_from_max'] ,data['RSST'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax4.plot(data['hours_from_max'] ,data['SHRD'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=4)
for i,ax in enumerate([ax2,ax4]):
    ax.set_ylabel(f"{'SHRD'} {variables['SHRD']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['SHRD'].min()-2,ships_select['SHRD'].max()+5)
    ax.set_xlabel('Time [Hours]', color=(0.45, 0.45, 0.45), fontsize = 9)
for i,ax in enumerate([ax1,ax3]):
    ax.set_ylabel(f"{'RSST'} {variables['RSST']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['RSST'].min()-1,ships_select['RSST'].max()+1)
ax1.set_title(f'Rapid Intensification Cases' , color="#4361ee", pad=5, fontsize=11,fontweight='bold')
ax3.set_title(f'Non Rapid Intensification Cases', color="#e09f3e", pad=5, fontsize=11,fontweight='bold')
plt.savefig(path_fig + f'SHIPS_FINAL_VAR_112.pdf',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)
plt.savefig(path_fig + f'SHIPS_FINAL_VAR_112.png',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)

# %% CONDICIONES 2
fig = plt.figure(figsize=(16,6),facecolor='none',edgecolor='none')
gs = GridSpec(2,2,wspace=.03,hspace=.06)#,height_ratios=[1,1,1,0.05])#,width_ratios=[1,0.05])#,height_ratios=[1,1,0.05])
ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])
ax3,ax4 = fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])
for i,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.spines['top'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['right'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.set_xlim(-100,+100)
    ax.grid(True,linestyle=':', color='gray', alpha=0.2)
    if i in [0,2]:
        ax.set_xticklabels([])  
        ax.set_xlabel('')
    if i in [2,3]:
        ax.set_yticklabels([])  
        ax.set_ylabel('')
    ax.axvline(0,color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
for TC in ships_select['ATCF_ID'].unique():
    data = ships_select[ships_select['ATCF_ID']==TC][['paso','ATCF_ID','TLON','TLAT','VMAX','VMPI','SHRD','RSST','FINAL_O_INTENSITY','MNCDPC','BASIN','hours_from_max']]
    intensity = data['FINAL_O_INTENSITY'].iloc[0]
    case = data['MNCDPC'].iloc[0]>=4
    basin = data['BASIN'].iloc[0]
    linestyle = '-' if case else ':'
    marker = 'o' if case else 'v'
    if intensity == 'RI':
        ax1.plot(data['hours_from_max'] ,data['VMAX'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax2.plot(data['hours_from_max'] ,data['VMPI'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
    if intensity == 'NI':
        ax3.plot(data['hours_from_max'] ,data['VMAX'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax4.plot(data['hours_from_max'] ,data['VMPI'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
for i,ax in enumerate([ax1,ax3]):
    ax.set_ylabel(f"{'VMAX'} {variables['VMAX']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['VMAX'].min()-5,ships_select['VMAX'].max()+10)
for i,ax in enumerate([ax2,ax4]):
    ax.set_ylabel(f"{'VMPI'} {variables['VMPI']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['VMPI'].min()-5,ships_select['VMPI'].max()+10)
    ax.set_xlabel('Time [Hours]', color=(0.45, 0.45, 0.45), fontsize = 9)
ax1.set_title(f'Rapid Intensification Cases' , color="#4361ee", pad=5, fontsize=11,fontweight='bold')
ax3.set_title(f'Non Rapid Intensification Cases', color="#e09f3e", pad=5, fontsize=11,fontweight='bold')
plt.savefig(path_fig + f'SHIPS_FINAL_VAR_212.pdf',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)
plt.savefig(path_fig + f'SHIPS_FINAL_VAR_212.png',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)

#%% CONDICIONES FULL
fig = plt.figure(figsize=(12,10),facecolor='none',edgecolor='none')
gs = GridSpec(4,2,wspace=.07,hspace=.16)#,height_ratios=[1,1,1,0.05])#,width_ratios=[1,0.05])#,height_ratios=[1,1,0.05])
ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])
ax3,ax4 = fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])
ax5,ax6 = fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])
ax7,ax8 = fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1])
for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
    ax.spines['top'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['right'].set_color((0.45, 0.45, 0.45))#.set_visible(False)
    ax.spines['left'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=9, colors=(0.45, 0.45, 0.45))
    ax.set_xlim(-100,100)
    ax.grid(True,linestyle=':', color='gray', alpha=0.2)
    if i in [0,1,2,3,4,5]:
        ax.set_xticklabels([])  
        ax.set_xlabel('')
    if i in [2,3,5,7]:
        ax.set_yticklabels([])  
        ax.set_ylabel('')
    ax.axvline(0,color = (0.45, 0.45, 0.45),alpha=.5,linestyle='dashed')
for TC in ships_select['ATCF_ID'].unique():
    data = ships_select[ships_select['ATCF_ID']==TC][['paso','ATCF_ID','TLON','TLAT','VMAX','VMPI','SHRD','RSST','FINAL_O_INTENSITY','MNCDPC','BASIN','hours_from_max']]
    intensity = data['FINAL_O_INTENSITY'].iloc[0]
    case = data['MNCDPC'].iloc[0]>=4
    basin = data['BASIN'].iloc[0]
    linestyle = '-' if case else ':'
    marker = 'o' if case else 'v'
    if intensity == 'RI':
        ax1.plot(data['hours_from_max'] ,data['RSST'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax2.plot(data['hours_from_max'] ,data['VMPI'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax5.plot(data['hours_from_max'] ,data['SHRD'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax7.plot(data['hours_from_max'] ,data['VMAX'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
    if intensity == 'NI':
        ax3.plot(data['hours_from_max'] ,data['RSST'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax4.plot(data['hours_from_max'] ,data['VMPI'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax6.plot(data['hours_from_max'] ,data['SHRD'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
        ax8.plot(data['hours_from_max'] ,data['VMAX'],linestyle=linestyle,color=colores.loc[TC]['color'],alpha=.4,marker=marker,markersize=3)
for i,ax in enumerate([ax1,ax3]):
    ax.set_ylabel(f"{'RSST'} {variables['RSST']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['RSST'].min()-1,ships_select['RSST'].max()+1)
for i,ax in enumerate([ax2,ax4]):
    ax.set_ylabel(f"{'VMPI'} {variables['VMPI']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['VMPI'].min()-5,ships_select['VMPI'].max()+10)
for i,ax in enumerate([ax5,ax6]):
    ax.set_ylabel(f"{'SHRD'} {variables['SHRD']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['SHRD'].min()-2,ships_select['SHRD'].max()+5)
for i,ax in enumerate([ax7,ax8]):
    ax.set_ylabel(f"{'VMAX'} {variables['VMAX']}",color = (0.45, 0.45, 0.45), fontsize=10) if i == 0 else ax.set_yticklabels([])
    ax.set_ylim(ships_select['VMAX'].min()-5,ships_select['VMAX'].max()+10)
    ax.set_xlabel('Time [Hours]', color=(0.45, 0.45, 0.45), fontsize = 9)
ax1.set_title(f'Rapid Intensification Cases' , color="#4361ee", pad=5, fontsize=11,fontweight='bold')
ax3.set_title(f'Non Rapid Intensification Cases', color="#e09f3e", pad=5, fontsize=11,fontweight='bold')

for ax, etiqueta in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 
                        ['a', 'c', 'b', 'd','e', 'f','g','h']):
    ax.text(0.01, 1.115, f'({etiqueta})', 
            transform=ax.transAxes,
            color = (0.45, 0.45, 0.45),
            fontsize=12, fontweight='bold', va='top', ha='left')

RI_LEG = fig.legend(RI_lines, RI_names,
                   bbox_to_anchor=(.98, .89),frameon=False,
                   labelspacing=2.55,
                   handlelength=2,loc='upper center', ncol=1)
NI_LEG = fig.legend(NI_lines, NI_names,
                   bbox_to_anchor=(1.11, .89),frameon=False,
                   labelspacing=2.18,
                   handlelength=2,loc='upper center', ncol=1)

for leg in [RI_LEG,NI_LEG]:
    for text in leg.get_texts():
        text.set_color((0.45, 0.45, 0.45))
        text.set_fontsize(8)

plt.savefig(path_fig + f'SHIPS_FINAL_VAR_ALL.pdf',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)
plt.savefig(path_fig + f'SHIPS_FINAL_VAR_ALL.png',pad_inches=0.1,bbox_inches='tight',dpi=300,transparent=True)
#%% GRAFICOS EXPLORATORIOS
for TC in ships_select['ATCF_ID'].unique():
    data0 = ships_select[ships_select['ATCF_ID']==TC]
    date_max = data0.loc[data0['VMAX'].idxmax()]['paso']
    for i,var in enumerate(variables.keys()):
        try:
            data = data0[[var,'paso']]
            fig = plt.figure(figsize=(6, 3),facecolor='w',edgecolor='w')
            ax1=fig.add_subplot(111)
            ax2 = ax1.secondary_xaxis('bottom')
            for ax in [ax1,ax2]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color((0.45, 0.45, 0.45))
                ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
                ax.tick_params(axis='x', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))
                ax.tick_params(axis='y', which='major', labelsize=7.5, colors=(0.45, 0.45, 0.45))

            ax1.plot(data['paso'],data[var])
            ax2.set_xlabel('Step\nDate',color=(0.45, 0.45, 0.45), fontsize = 10)
            ax1.set_ylabel(f'{var} {variables[var]}',color = (0.45, 0.45, 0.45), fontsize=10)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(data['paso'])
            ax2.set_xticklabels(data.index.strftime('%H %m-%d'), rotation=270, ha='center')
            ax2.tick_params(axis='x', pad=15, direction='out', length=2.5)
            ax1.axvline(x=date_max, color=(0.45, 0.45, 0.45), linestyle=':', linewidth=1)
            plt.savefig(path_fig + 'BASIC/'+ f'{TC}_{var}.png',pad_inches=0.1,bbox_inches='tight',dpi=300)
            plt.close()
        except:
            pass


#%% CASO RARO
da= ships_select[ships_select['ATCF_ID'] == 'AL062022']#[['TLON','TLAT']].reset_index(drop=True)
fig = plt.figure(figsize=(19,4),facecolor='none',edgecolor='none')
ax1=fig.add_subplot(111,projection=crs.PlateCarree(central_longitude=0))
ax1.coastlines(color='gray')
ax1.plot(centrar_longitud(da['TLON']),da['TLAT'])
ax1.set_xlim(centrar_longitud(AL_L))
# %%
