#!/usr/bin/env python
# coding: utf-8


"""
Codigo para analisi espectral de series temporales de caractetisticas convectivas.

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.4"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__methods__:
__comands__:
__changues__:
    - [2025-05-26][Duvan]: Primera version del codigo.

"""
#%% MODULOS
from seaborn import heatmap,scatterplot
#from google.colab import drive # CAMBIO DRIVE
from collections import defaultdict
from matplotlib import pyplot as plt
from seaborn import histplot,kdeplot,move_legend
from scipy.stats import mannwhitneyu
from pandas import read_csv,DataFrame
from statsmodels.tsa.stattools import adfuller
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import (mean,unique,fft,abs,arange,any,trapezoid,std,
                   median,max,nan,full,linspace,nanmin,nanmax,
                   nanmedian,nansum,shape,nanmean,
                   zeros_like,full_like,std)
#%% PATHS
#drive.mount('/content/drive') # CAMBIO DRIVE
path_general = '/content/drive/MyDrive/cursos_proyectos/Huracanes/'
path_img     = path_general + 'Img/'
path_general = '/home/dunievesr/Dropbox/UNAL/TESIS/Final_codes/' # CAMBIO DRIVE
path_img     = '/home/dunievesr/Datos/TESIS/FIG/' # CAMBIO DRIVE

#%%FUNCIONES
def get_fourier(series_temp,T=1/6):
    T = 1/6
    t = arange(0,len(series_temp)*T,T)
    y = series_temp
    # Perform Fourier Transform
    signal_fft = fft.fft(y)
    freq = fft.fftfreq(len(y), d=T)

    power_spectrum = (abs(signal_fft)**2) /len(y)
    # Remove negative frequencies
    positive_freq = freq[:len(freq)//2][1:]
    period = 1/positive_freq
    positive_power = power_spectrum[:len(freq)//2][1:]
#   positive_power = positive_power/np.sum(positive_power)
    return period,positive_power

def get_fourier_with_integral(series_temp, T=1/6, p0=None, p1=None):
    # Time vector (not explicitly used for Fourier, but included for completeness)
    t = arange(0, len(series_temp) * T, T)
    y = series_temp

    # Perform Fourier Transform
    signal_fft = fft.fft(y)
    freq = fft.fftfreq(len(y), d=T)
    power_spectrum = (abs(signal_fft)**2)/len(y)

    # Remove negative frequencies
    positive_freq = freq[:len(freq) // 2][1:]
    period = 1 / positive_freq
    positive_power = power_spectrum[:len(freq) // 2][1:]

    # Initialize integrated_power to None
    integrated_power = None

    # Compute the integral of power between p0 and p1 if provided
    if p0 is not None and p1 is not None:
        # Convert periods to frequency range
        f_min = 1 / p1  # Maximum period p1 corresponds to minimum frequency
        f_max = 1 / p0  # Minimum period p0 corresponds to maximum frequency

        # Identify indices within the frequency range
        indices = (positive_freq >= f_min) & (positive_freq <= f_max)

        # Perform numerical integration using the trapezoidal rule
        if any(indices):  # Ensure there are valid points for integration
            integrated_power = trapezoid(positive_power[indices], positive_freq[indices])
        else:
            #print('no data')
            integrated_power = 0.0  # No valid data points in the range

    return period, positive_power, integrated_power

def extract_periods(period_key):
    p0, p1 = map(float, period_key.split('-'))
    return p0, p1

def select_variability(df, columnas_seleccionadas, window=48):
    # Función que se aplicará a cada grupo
    def aplicar_a_grupo(grupo):
        grupo_resultado = grupo.copy()
        for columna in columnas_seleccionadas:
            serie_original = grupo[columna].interpolate(method='linear', axis=0).ffill().bfill()
            rolling_mean = serie_original.rolling(6*window, center=True, min_periods=6).mean()
            detrend = serie_original - rolling_mean
            grupo_resultado[columna] = detrend/detrend.std()
        return grupo_resultado
    
    # Aplicamos la función a cada grupo y concatenamos los resultados
    return df.groupby('ATCF_ID', group_keys=False).apply(aplicar_a_grupo)

def filter_by_period(df, columnas_seleccionadas, p0, p1, group_col='ATCF_ID', T=1/6):
    """
    Filtra series temporales utilizando la transformada de Fourier para mantener solo 
    componentes dentro de un rango específico de periodos.
    
    Args:
        df: DataFrame con los datos
        columnas_seleccionadas: Lista de columnas a filtrar
        p0: Periodo mínimo a mantener
        p1: Periodo máximo a mantener
        group_col: Columna para agrupar los datos (por defecto 'ATCF_ID')
        T: Intervalo de tiempo entre muestras (por defecto 1/6)
        
    Returns:
        DataFrame filtrado con las mismas dimensiones que el original
    """
    # Creamos una copia para no modificar el DataFrame original
    df_resultado = df.copy()
    
    # Convertimos a frecuencias
    f_min = 1 / p1  # Periodo mayor corresponde a frecuencia menor
    f_max = 1 / p0  # Periodo menor corresponde a frecuencia mayor
    
    # Función para aplicar a cada grupo
    def aplicar_filtro_fourier(grupo):
        grupo_resultado = grupo.copy()
        
        for columna in columnas_seleccionadas:
            if columna in grupo.columns:
                # Obtenemos la serie
                y = grupo[columna].values
                
                # Aplicamos FFT
                y_fft = fft.fft(y)
                freqs = fft.fftfreq(len(y), d=T)

                # Creamos una máscara de frecuencias que queremos mantener
                # (incluimos frecuencias positivas y negativas simétricamente)
                mask = (abs(freqs) >= f_min) & (abs(freqs) <= f_max)

                # Aplicamos la máscara (ponemos a cero las frecuencias fuera del rango)
                y_fft_filtered = y_fft.copy()
                y_fft_filtered[~mask] = 0

                # Transformada inversa para volver al dominio del tiempo
                y_filtered = fft.ifft(y_fft_filtered).real

                # Actualizamos la columna en el grupo resultado
                grupo_resultado[columna] = y_filtered
        
        return grupo_resultado
    
    # Aplicamos la función a cada grupo
    resultado = df.groupby(group_col, group_keys=False).apply(aplicar_filtro_fourier)
    
    return resultado

def scale_by_max(df, columnas):
    df_scaled = df.copy()
    for columna in columnas:
        max_abs = df[columna].abs().max()
        df_scaled[columna] = df[columna] / max_abs
    return df_scaled

def scale_by_max_grouped(df, columnas):
    def escalar_grupo(grupo):
        grupo_escalado = grupo.copy()
        for columna in columnas:
            max_abs = grupo[columna].abs().max()
            grupo_escalado[columna] = grupo[columna] / max_abs
        return grupo_escalado
    
    return df.groupby('ATCF_ID', group_keys=False).apply(escalar_grupo)
#%% PALETAS
color_min = '#4361ee'  # Azul
color_max = '#540b0e'  # Rojo
custom_cmap = LinearSegmentedColormap.from_list('custom', [color_min, 'whitesmoke', color_max])
#%%DATOS
df = read_csv(path_general+'datos.csv',index_col=0)
df = df[df['hours_from_max'].between(-100,0)]
df.index = df.hours_from_max
ri_labels = unique(df[df['RI']==1]['ATCF_ID'])
nori_labels = unique(df[df['RI']==0]['ATCF_ID'])
#%% SERIES SIN TENDENCIA RI
series_ri = [];means_ri = []
for i in range(len(ri_labels)):
    label_ri_temp = ri_labels[i]
    series_ri_interp = df[df['ATCF_ID'] == label_ri_temp]['ht_area'].interpolate(method='linear', axis=0).ffill().bfill()
    # Cálculo de la media
    means_ri.append(mean(series_ri_interp))
    # Detrend usando ventana móvil
    detrended_ri = series_ri_interp - series_ri_interp.rolling(6*48, center=True, min_periods=6).mean()# Nos quedamos con la variabilidad menor a 48H  
    #series_ri.append(detrended_ri / detrended_ri.std())
    series_ri.append(detrended_ri)
#%% SERIES SIN TENDENCIA NI
series_nori = []
means_nori = []
for i in range(len(nori_labels)):
    label_nori_temp = nori_labels[i]
    series_nori_interp = df[df['ATCF_ID']==label_nori_temp]['ht_area'].interpolate(method='linear', axis=0).ffill().bfill()
    means_nori.append(mean(series_nori_interp))
    
    detrended_nori = series_nori_interp - series_nori_interp.rolling(6*48, center=True, min_periods=6).mean()# Nos quedamos con la variabilidad menor a 48H  
    #series_nori.append(detrended_nori / detrended_nori.std())
    series_nori.append(detrended_nori)
#%% Estadisticos
print("Medias:"f" RI: {mean(means_ri)}"f" NI: {mean(means_nori)}")
#%% ESPECTROS
p0 = 3; p1 = 24
#%% RI
pows_ri = []
for i in range(len(series_ri)):
    p_temp,pow_temp,integral = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=p0,p1=p1)
#    pow_0_6 = np.sum(pow_temp[np.where((p_temp>=p0)&(p_temp<p1))[0]])
    pows_ri.append(integral)
#%% NI
pows_nori = []
for i in range(len(series_nori)):
    p_temp,pow_temp,integral = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=p0,p1=p1)
#    pow_0_6 = np.sum(pow_temp[np.where((p_temp>=p0)&(p_temp<p1))[0]])
    pows_nori.append(integral)
#%% ESPECTROS INTEGRADOS
results_df = DataFrame(columns=['period_diff', 'period_diff_half', 'median_diff', 'percent_diff', 'significativo'])
p_range = range(1, 48,1)
# Ciclos para calcular con diferentes períodos
for p0 in p_range:
    for p1 in p_range:
        if p0 <= p1:  # Solo cuando p0 es menor que p1
            period_key = f"{p0}-{p1}"
            period_diff = p1 - p0  # Calculamos la diferencia entre períodosperiod_diff_half
            period_diff_half = (p1 + p0) / 2  # Calculamos la mitad de la diferencia

            # Calcular para series_ri
            pows_ri = []
            for series in series_ri:
                p_temp, pow_temp, integral = get_fourier_with_integral(
                    series/std(series),
                    p0=p0,
                    p1=p1
                )
                pows_ri.append(integral)

            # Calcular para series_nori
            pows_nori = []
            for series in series_nori:
                p_temp, pow_temp, integral = get_fourier_with_integral(
                    series/std(series),
                    p0=p0,
                    p1=p1
                )
                pows_nori.append(integral)

            pows_ri = pows_ri / nanmax(pows_ri) #Escalado
            pows_nori = pows_nori / nanmax(pows_nori) #Escalado

            # Añadir prueba de significancia
            _, p_valor = mannwhitneyu(pows_ri, pows_nori, alternative='two-sided')
            es_significativo = p_valor < 0.05

            # Calcular las medianas
            median_ri = nanmedian(pows_ri)
            median_nori = nanmedian(pows_nori)

            # Calcular las diferencias
            median_diff = median_ri - median_nori
            percent_diff = abs(nansum(pows_ri) / (nansum(pows_ri)+nansum(pows_nori))) * 100 if median_nori != 0 else nan

            # Agregar al DataFrame
            results_df.loc[period_key] = [period_diff, period_diff_half, median_diff, percent_diff,es_significativo]

# Ordenar el DataFrame por la diferencia de períodos
results_df = results_df.sort_values('period_diff')
#%% AJUSTE DF:
results_df.loc[~results_df['significativo'], 'Type'] = 'NS'
results_df.loc[(results_df['significativo'])&(results_df['median_diff']>0), 'Type'] = 'RIS'
results_df.loc[(results_df['significativo'])&(results_df['median_diff']<0), 'Type'] = 'NIS'
#%% HeatMap
results_df['p0'] = results_df.index.map(lambda x: extract_periods(x)[0])
results_df['p1'] = results_df.index.map(lambda x: extract_periods(x)[1])
# Encontrar el período mínimo y máximo global
min_period = min([results_df['p0'].min(), results_df['p1'].min()])
max_period = max([results_df['p0'].max(), results_df['p1'].max()])
# Crear un rango de períodos completo
periods = arange(min_period, max_period + 1,1).astype('int')
n_periods = len(periods)
# Creamos la matriz para el pcolormesh
median_diff_matrix = full((n_periods, n_periods), nan)  # Inicializamos con NaN
# Llenamos la matriz completa donde p0 < p1
for i, p0 in enumerate(periods):
    for j, p1 in enumerate(periods):
        if p0 <= p1:
            # Buscamos este par p0-p1 en el DataFrame original
            match = results_df[(results_df['p0'] == p0) & (results_df['p1'] == p1)]
            if not match.empty:
                median_diff_matrix[j, i] = match['median_diff'].values[0]
            else:
                # Si no existe el valor, podríamos interpolarlo o dejarlo como NaN
                median_diff_matrix[j, i] = nan

annot_matrix =  full_like(median_diff_matrix, '', dtype=str)
for i, p0 in enumerate(periods):
    for j, p1 in enumerate(periods):
        if p0 <= p1:
            match = results_df[(results_df['p0'] == p0) & (results_df['p1'] == p1)]
            if not match.empty and match['significativo'].values[0]:
                annot_matrix[j, i] = '*'

# Creamos una máscara para los valores donde p0 >= p1
mask = zeros_like(median_diff_matrix, dtype=bool)
for i in range(n_periods):
    for j in range(n_periods):
        if periods[i] >= periods[j]:
            mask[j, i] = True

#%% FIGURA
fig = plt.figure(figsize=(8, 8), facecolor='None', edgecolor='None')
ax = fig.add_subplot(111)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))

ax.tick_params(axis='both', which='major', labelsize=7.5,colors=(0.45, 0.45, 0.45))

tick_labels = [str(p).zfill(2) for p in periods]
heatmap(median_diff_matrix,
            #mask=mask,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cmap=custom_cmap,
            center=0,
            annot=annot_matrix,  # Añadir matriz de anotaciones
            fmt='',  # Formato vacío para texto
            cbar_kws={'label': 'Difference of Medians',
                      'location': 'left',
                      'ticks': linspace(nanmin(median_diff_matrix),
                                       nanmax(median_diff_matrix),
                                       20)
                     },
            ax=ax)
#ax.set_title('Diferencia de Medianas')
ax.set_xlabel('Initial Period [Hours]',color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_ylabel('Final Period [Hours]',color=(0.45, 0.45, 0.45), fontsize=10)

ax.set_xlim(0, len([p for p in periods if p <= 26]))

ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors=(0.45, 0.45, 0.45), width=0.5)
cbar.set_label('Difference of Medians', color=(0.45, 0.45, 0.45))
plt.savefig(path_img + 'Diff_mediasn.png',pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_img + 'Diff_mediasn.pdf',pad_inches=0.1,bbox_inches='tight',dpi=250)
#%% FULL FIGURE 2
pows_ri_2_4 = []
pows_ri_6_18 = []
pows_ri_7_30 = []
pows_ri_11_31 = []
for i in range(len(series_ri)):
    p_temp,pow_temp,integral_2_4 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]) ,p0=2,p1=4)
    p_temp,pow_temp,integral_6_18 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=6,p1=18)
    p_temp,pow_temp,integral_7_30 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=7,p1=30)
    p_temp,pow_temp,integral_11_31 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=11,p1=31)
    pows_ri_2_4.append(integral_2_4)
    pows_ri_6_18.append(integral_6_18)
    pows_ri_7_30.append(integral_7_30)
    pows_ri_11_31.append(integral_11_31)

pows_nori_2_4 = []
pows_nori_6_18 = []
pows_nori_7_30 = []
pows_nori_11_31 = []
for i in range(len(series_nori)):
    p_temp,pow_temp,integral_2_4 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]) ,p0=2,p1=4)
    p_temp,pow_temp,integral_6_18 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=6,p1=18)
    p_temp,pow_temp,integral_7_30 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=7,p1=30)
    p_temp,pow_temp,integral_11_31 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=11,p1=31)
    pows_nori_2_4.append(integral_2_4)
    pows_nori_6_18.append(integral_6_18)
    pows_nori_7_30.append(integral_7_30)
    pows_nori_11_31.append(integral_11_31)

type_colors = {'NS':(0.45, 0.45, 0.45,0.05),
               'RIS': (0.329, 0.043, 0.055, 0.7),
               'NIS': (0.263, 0.380, 0.933, 0.7)}

fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='None', edgecolor='None')

# Estilo de ejes
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=8.5, colors=(0.45, 0.45, 0.45))
ax.set_xlabel('Half Period [Hours]', color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_ylabel('Difference of medians of the spectrum', color=(0.45, 0.45, 0.45), fontsize=10)
ax.axhline(y=0, color=(0.45, 0.45, 0.45), linestyle='--', alpha=0.7)

# Filtro por period_diff > 1
filtered = results_df[results_df['period_diff'] > 1]

# Graficar por tipo
for tipo, grupo in filtered.groupby("Type"):
    ax.scatter(grupo["period_diff_half"], grupo["median_diff"],
               s=(abs(grupo["percent_diff"] // 3) )**2 ,
               color=type_colors[tipo],
               label=tipo, edgecolors='none')

legend_elements = [
    Patch(facecolor=(0.329, 0.043, 0.055, 0.7), label='Significant Rapid Intensification'),
    Patch(facecolor=(0.263, 0.380, 0.933, 0.7), label='No Rapid Significant Intensification')
]
legend = ax.legend(handles=legend_elements, loc='upper center',ncol=2,
                   bbox_to_anchor=(0.5, 1.1), 
                   frameon=False, fontsize=8.5, title_fontsize=9)

for text in legend.get_texts():
  text.set_color((0.45, 0.45, 0.45))
  text.set_fontsize(10)
    
ax.set_ylim(-.3,.45)
    
# Insertar axes 1
inset_ax1 = inset_axes(ax, width="25%", height="25%",
                      loc='upper right',
                      bbox_to_anchor=(0, -0.2, 1, 1),
                      bbox_transform=ax.transAxes)
inset_ax1.patch.set_facecolor('none')

for spine in ['top', 'right']:
    inset_ax1.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    inset_ax1.spines[spine].set_color((0.45, 0.45, 0.45))

inset_ax1.tick_params(axis='both', which='major', labelsize=6.5,colors=(0.45, 0.45, 0.45))

histplot(x=pows_ri_11_31, color="#540b0e", kde=False,ax=inset_ax1,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_ri_11_31, color="#540b0e", ax=inset_ax1, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3
histplot(x=pows_nori_11_31, color="#4361ee", kde=False,ax=inset_ax1,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_nori_11_31, color="#4361ee", ax=inset_ax1, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3

inset_ax1.axvline(median(pows_ri_11_31),color='#540b0e')
inset_ax1.axvline(median(pows_nori_11_31),color='#4361ee')
inset_ax1.set_ylabel(f'Density', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax1.set_xlabel(f'Power', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax1.set_title('Power Distribution Area of\nHot Towers'+ ' ' +f'over {11}-{31} Hours' , fontsize=7, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))


ax.annotate('',
            xy=(0.74, .8), xycoords='axes fraction',
            xytext=(21, .25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


ax.annotate('',
            xy=(0.72, .53), xycoords='axes fraction',
            xytext=(21, .25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


# Insertar axes 2
inset_ax2 = inset_axes(ax, width="25%", height="25%",
                      loc='upper right',
                      bbox_to_anchor=(0, -0.685, 1, 1),
                      bbox_transform=ax.transAxes)
inset_ax2.patch.set_facecolor('none')

for spine in ['top', 'right']:
    inset_ax2.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    inset_ax2.spines[spine].set_color((0.45, 0.45, 0.45))

inset_ax2.tick_params(axis='both', which='major', labelsize=6.5,colors=(0.45, 0.45, 0.45))

histplot(x=pows_ri_7_30, color="#540b0e", kde=False,ax=inset_ax2,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_ri_7_30, color="#540b0e", ax=inset_ax2, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3
histplot(x=pows_nori_7_30, color="#4361ee", kde=False,ax=inset_ax2,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_nori_7_30, color="#4361ee", ax=inset_ax2, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3

inset_ax2.axvline(median(pows_ri_7_30),color='#540b0e')
inset_ax2.axvline(median(pows_nori_7_30),color='#4361ee')
inset_ax2.set_ylabel(f'Density', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax2.set_xlabel(f'', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax2.set_title('Power Distribution Area of\nHot Towers'+ ' ' +f'over {7}-{30} Hours' , fontsize=7, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))

ax.annotate('',
            xy=(0.74, .33), xycoords='axes fraction',
            xytext=(18.5, 0.25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


ax.annotate('',
            xy=(0.72, .065), xycoords='axes fraction',
            xytext=(18.5, 0.25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


# Insertar axes 3
inset_ax3 = inset_axes(ax, width="25%", height="25%",
                      loc='upper right',
                      bbox_to_anchor=(-0.35, -0.685, 1, 1),
                      bbox_transform=ax.transAxes)
inset_ax3.patch.set_facecolor('none')

for spine in ['top', 'right']:
    inset_ax3.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    inset_ax3.spines[spine].set_color((0.45, 0.45, 0.45))
    
inset_ax3.tick_params(axis='both', which='major', labelsize=6.5,colors=(0.45, 0.45, 0.45))

histplot(x=pows_ri_6_18, color="#540b0e", kde=False,ax=inset_ax3,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_ri_6_18, color="#540b0e", ax=inset_ax3, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3
histplot(x=pows_nori_6_18, color="#4361ee", kde=False,ax=inset_ax3,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_nori_6_18, color="#4361ee", ax=inset_ax3, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3


inset_ax3.axvline(median(pows_ri_6_18),color='#540b0e')
inset_ax3.axvline(median(pows_nori_6_18),color='#4361ee')
inset_ax3.set_ylabel(f'Density', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax3.set_xlabel(f'', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax3.set_title('Power Distribution Area of\nHot Towers'+ ' ' +f'over {6}-{18} Hours' , fontsize=7, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))

ax.annotate('',
            xy=(0.39, .33), xycoords='axes fraction',
            xytext=(12, 0.29), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


ax.annotate('',
            xy=(0.363, .065), xycoords='axes fraction',
            xytext=(12, 0.29), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))

plt.savefig(path_img + 'Diff_medias_scatter_half.png',pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_img + 'Diff_medias_scatter_half.pdf',pad_inches=0.1,bbox_inches='tight',dpi=250)

#%% FILTROS
cols_vars = ['ht_count','ht_area','rb_area','rb_bt_mean','rb_bt_std']
df_48 = select_variability(df.drop(columns='hours_from_max').sort_values(by=['ATCF_ID','hours_from_max']),cols_vars)
filtro_11_31 = filter_by_period(df_48, cols_vars, 8,8.4 )
filtro_11_31_scaled = scale_by_max_grouped(filtro_11_31, cols_vars)
filtro_11_31_scaled
var = 'ht_count'
data = filtro_11_31_scaled[[var,'ATCF_ID','RI']]
fig, ax = plt.subplots(figsize=(10, 2), facecolor='None', edgecolor='None')

# Estilo de ejes
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=8.5, colors=(0.45, 0.45, 0.45))
ax.set_xlabel('', color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_ylabel(var, color=(0.45, 0.45, 0.45), fontsize=10)
(data.query('RI==1')
 .groupby(level="hours_from_max")[var]
 .mean().rolling(6,center=True,min_periods=6)
 .mean().plot(ax=ax,color=(0.329, 0.043, 0.055, 0.7)))
((data.query('RI==0')*-1)
 .groupby(level="hours_from_max")[var]
 .mean().rolling(6,center=True,min_periods=6)
 .mean().plot(ax=ax,color=(0.263, 0.380, 0.933, 0.7)))
for tc in data['ATCF_ID'].unique():
    data_tc_i = data.query(f"ATCF_ID == '{tc}'")
    if data_tc_i['RI'].sum()>0:
        data_tc_i[var].plot(ax=ax,color=(0.329, 0.043, 0.055, 0.05))
    else:
        (-data_tc_i[var]).plot(ax=ax,color=(0.263, 0.380, 0.933, 0.05))
        
#asi no sirve necesito la fecha y hacerlo por uracan filtrando, donde se vea claramente la señal
#NEcesito coservar la fecha
# Que no se pierda para depsues poner al serie filtrada y ver si es un forzamiento del ciclo diurno
#for i in range(0,100,18):
#    ax.axvline(-i)
#%% 
var = 'ht_count'
data = filtro_11_31_scaled[[var,'ATCF_ID','RI']]

fig, ax1 = plt.subplots(figsize=(8, 4), facecolor='None', edgecolor='None')

ax2,ax3 = ax1.twinx(),ax1.twinx()
ax3.spines.right.set_position(("axes", 1.1))

(df.query('ATCF_ID == "AL062018"')[var]
 .rolling(6,center=True,min_periods=6).mean()
 .rolling(6,center=True,min_periods=6).mean()
 .plot(ax=ax1,color='#0066FF',alpha=.6))
(df.query('ATCF_ID == "EP202018"')[var]
 .rolling(6,center=True,min_periods=6).mean()
 .rolling(6,center=True,min_periods=6).mean()
 .plot(ax=ax2,color='#2000FF',alpha=.6))
(df.query('ATCF_ID == "WP252017"')[var]
 .rolling(6,center=True,min_periods=6).mean()
 .rolling(6,center=True,min_periods=6).mean()
 .plot(ax=ax3,color='#E000FF',alpha=.6))

for i,ax in enumerate([ax1,ax2,ax3]):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#0066FF')
    ax.spines['right'].set_color((0.45, 0.45, 0.45))
    ax.spines['bottom'].set_color((0.45, 0.45, 0.45))
    ax.tick_params(axis='x', which='major', labelsize=10, colors=(0.45, 0.45, 0.45))
    ax.tick_params(axis='y', which='major', labelsize=10, colors=(0.45, 0.45, 0.45))
    ax.set_xlabel("", color='#2000FF')
    
ax1.tick_params(axis='y', colors='#0066FF',labelsize=10)
ax2.tick_params(axis='y', colors='#2000FF',labelsize=10)
ax3.tick_params(axis='y', colors='#E000FF',labelsize=10)

ax1.set_ylabel("NHT AL062018", color='#0066FF')
ax2.set_ylabel("NHT EP202018", color='#2000FF')
ax3.set_ylabel("NHT WP252017", color='#E000FF')

ax1.spines['left'].set_color('#0066FF')
ax2.spines['right'].set_color('#2000FF')
ax3.spines['right'].set_color('#E000FF')

    
band_start = 21 * (-100 // 21)
ticks_positions = list(range(int(band_start), 0, 21))
ax1.set_xticks(ticks_positions, minor=True)
ax1.tick_params(axis='x', which='minor', direction='in', length=10, width=1, color=(0.45, 0.45, 0.45,.5))

ax.axvline(-30, color= '#0066FF', linestyle='--', alpha=0.4)
ax.axvline(-12, color= '#2000FF', linestyle='--', alpha=0.4)
ax.axvline(-24, color= '#E000FF', linestyle='--', alpha=0.4)


inset_ax1 = inset_axes(ax1, width="100%", height="60%",
                      loc='upper right',
                      bbox_to_anchor=(0.009, .65, 1, 1),
                      bbox_transform=ax.transAxes)
inset_ax1.patch.set_facecolor('none')

for spine in ['top', 'right']:
    inset_ax1.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    inset_ax1.spines[spine].set_color((0.45, 0.45, 0.45))

inset_ax1.tick_params(axis='both', which='major', labelsize=10,colors=(0.45, 0.45, 0.45))



inset_ax1.set_xlabel('', color=(0.45, 0.45, 0.45), fontsize=10)
inset_ax1.set_ylabel(var, color=(0.45, 0.45, 0.45), fontsize=10)
(data.query('RI==1')
 .groupby(level="hours_from_max")[var]
 .mean().rolling(6,center=True,min_periods=6)
 .mean().plot(ax=inset_ax1,color=(0.329, 0.043, 0.055, 0.7)))
(data.query('RI==0')
 .groupby(level="hours_from_max")[var]
 .mean().rolling(6,center=True,min_periods=6)
 .mean().plot(ax=inset_ax1,color=(0.263, 0.380, 0.933, 0.7)))
for tc in data['ATCF_ID'].unique():
    data_tc_i = data.query(f"ATCF_ID == '{tc}'")
    if data_tc_i['RI'].sum()>0:
        data_tc_i[var].plot(ax=inset_ax1,color=(0.329, 0.043, 0.055, 0.05))
    else:
        data_tc_i[var].plot(ax=inset_ax1,color=(0.263, 0.380, 0.933, 0.05))
inset_ax1.set_xlim(ax1.get_xlim())
inset_ax1.set_xticklabels([])
inset_ax1.set_ylabel('Number of Hot Towers (NHT)',size=9,color=(0.45, 0.45, 0.45))
inset_ax1.set_xlabel('',size=10,color=(0.45, 0.45, 0.45))

inset_ax1.set_xticks(ticks_positions, minor=True)
inset_ax1.tick_params(axis='x', which='minor', direction='in', length=10, width=1, color=(0.45, 0.45, 0.45,.5))

plt.savefig(path_img + 'Count_HT_cases.png',pad_inches=0.1,bbox_inches='tight',dpi=250)
plt.savefig(path_img + 'Count_HT_cases.pdf',pad_inches=0.1,bbox_inches='tight',dpi=250)

#%% VENTANA MOVIL ESPECTROS
p0 = 3; p1 = 48
pows_ri = []
for i in range(len(series_ri)):
    p_temp,pow_temp,integral = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=p0,p1=p1)
#    pow_0_6 = np.sum(pow_temp[np.where((p_temp>=p0)&(p_temp<p1))[0]])
    pows_ri.append(integral)
pows_nori = []
for i in range(len(series_nori)):
    p_temp,pow_temp,integral = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=p0,p1=p1)
#    pow_0_6 = np.sum(pow_temp[np.where((p_temp>=p0)&(p_temp<p1))[0]])
    pows_nori.append(integral)


# %%### ESPECTROS INTEGRADOS
results_rolling = DataFrame(columns=['window_center', 'median_diff', 'percent_diff', 'significativo'])

window_size_hours = 4  # horas - ajusta este valor
p_min = 1  # hora mínima a analizar
p_max = 48  # hora máxima a analizar

# Crear secuencia de centros de ventana
step_size = 0.1  
centers = arange(p_min + window_size_hours/2, 
                   p_max - window_size_hours/2, 
                   step_size)


# Ventana móvil
for center in centers:
    p0 = center - window_size_hours / 2
    p1 = center + window_size_hours / 2
    
    period_key = f"{round(p0,1)}-{round(p1,1)}"
    window_center = center
    
    # Calcular para series_ri
    pows_ri = []
    for series in series_ri:
        p_temp, pow_temp, integral = get_fourier_with_integral(
            series/std(series),
            p0=p0,
            p1=p1
        )
        pows_ri.append(integral)
    
    # Calcular para series_nori
    pows_nori = []
    for series in series_nori:
        p_temp, pow_temp, integral = get_fourier_with_integral(
            series/std(series),
            p0=p0,
            p1=p1
        )
        pows_nori.append(integral)
    
    pows_ri = pows_ri #/ nanmax(pows_ri)  # Escalado
    pows_nori = pows_nori #/ nanmax(pows_nori)  # Escalado
    
    # Añadir prueba de significancia
    _, p_valor = mannwhitneyu(pows_ri, pows_nori, alternative='two-sided')
    es_significativo = p_valor < 0.05
    
    # Calcular las medianas
    median_ri = nanmean(pows_ri)
    median_nori = nanmean(pows_nori)
    
    # Calcular las diferencias
    median_diff = median_ri - median_nori
    percent_diff = abs(nansum(pows_ri) / (nansum(pows_ri) + nansum(pows_nori))) * 100 if median_nori != 0 else nan
    
    # Agregar al DataFrame
    results_rolling.loc[period_key] = [window_center, median_diff, percent_diff, es_significativo]

# Ordenar el DataFrame por el centro de la ventana
results_rolling = results_rolling.sort_values('window_center')
results_rolling
#%% SELCCION DE ESPECTROS
results_rolling.loc[~results_rolling['significativo'], 'Type'] = 'NS'
results_rolling.loc[(results_rolling['significativo'])&(results_rolling['median_diff']>0), 'Type'] = 'RIS'
results_rolling.loc[(results_rolling['significativo'])&(results_rolling['median_diff']<0), 'Type'] = 'NIS'
df_roll_cut = results_rolling.query("Type!='NS'").sort_values('window_center')
results_rolling.dropna()
type_colors = {'NS':(0.45, 0.45, 0.45,0.05),
               'RIS': (0.329, 0.043, 0.055, 0.7),
               'NIS': (0.263, 0.380, 0.933, 0.7)}

fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='None', edgecolor='None')

# Estilo de ejes
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=8.5, colors=(0.45, 0.45, 0.45))
ax.set_xlabel('Center Windows', color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_ylabel('Difference of medians of the spectrum', color=(0.45, 0.45, 0.45), fontsize=10)
ax.axhline(y=0, color=(0.45, 0.45, 0.45), linestyle='--', alpha=0.7)

# Filtro por period_diff > 1
filtered = results_rolling.copy()

# Graficar por tipo
for tipo, grupo in filtered.groupby("Type"):
    ax.scatter(grupo["window_center"], grupo["median_diff"],
               s=(abs(grupo["percent_diff"] // 3) )**2 ,
               color=type_colors[tipo],
               label=tipo, edgecolors='none')

legend_elements = [
    Patch(facecolor=(0.329, 0.043, 0.055, 0.7), label='Significant Rapid Intensification'),
    Patch(facecolor=(0.263, 0.380, 0.933, 0.7), label='No Rapid Significant Intensification')
]
legend = ax.legend(handles=legend_elements, loc='upper center',ncol=2,
                   bbox_to_anchor=(0.5, 1.1), 
                   frameon=False, fontsize=8.5, title_fontsize=9)

for text in legend.get_texts():
  text.set_color((0.45, 0.45, 0.45))
  text.set_fontsize(10)

#%%