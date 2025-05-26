#!/usr/bin/env python
# coding: utf-8


"""
Codigo para analisi espectral de series temporales de caractetisticas convectivas _ 2.

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
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import (mean,unique,fft,abs,arange,any,trapezoid,std,
                   median,max,nan,full,linspace,nanmin,nanmax,
                   nanmedian,nansum,shape,array,
                   zeros_like,full_like,std)
#%% PATHS
#drive.mount('/content/drive') # CAMBIO DRIVE
path_general = '/content/drive/MyDrive/cursos_proyectos/Huracanes/'
path_img     = path_general + 'Img/'
path_general = '/home/dunievesr/Dropbox/UNAL/TESIS/Final_codes/' # CAMBIO DRIVE
path_img     = '/home/dunievesr/Datos/TESIS/FIG/' # CAMBIO DRIVE
#%% FUNCIONES
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

def analyze_specific_periods(series_ri, series_nori, T=1/6):
    """
    Analiza diferencias período por período específico
    Versión corregida que maneja series de diferentes longitudes
    """
    results_list = []
    
    # Definir períodos de interés (en horas)
    # Crear una lista logarítmica de períodos para tener buena cobertura
    periods_of_interest = arange(p0,p1+.1,.1)
    
    print(f"Analizando {len(periods_of_interest)} períodos específicos")
    print(f"Rango de períodos: {periods_of_interest.min():.2f} a {periods_of_interest.max():.2f} horas")
    
    # Para cada período específico de interés
    for target_period in periods_of_interest:
        target_freq = 1 / target_period
        
        # Obtener la potencia en este período para cada serie
        powers_ri = []
        powers_nori = []
        
        # Para RI
        for series in series_ri:
            # Calcular FFT para esta serie específica
            series_norm = series / std(series)
            signal_fft = fft.fft(series_norm)
            freq = fft.fftfreq(len(series_norm), d=T)
            positive_freq = freq[:len(freq) // 2][1:]
            power_spectrum = (abs(signal_fft)**2) / len(series_norm)
            positive_power = power_spectrum[:len(signal_fft) // 2][1:]
            
            # Encontrar la frecuencia más cercana a la objetivo
            if len(positive_freq) > 0:
                idx_closest = np.argmin(np.abs(positive_freq - target_freq))
                powers_ri.append(positive_power[idx_closest])
            else:
                powers_ri.append(0.0)
        
        # Para NoRI
        for series in series_nori:
            # Calcular FFT para esta serie específica
            series_norm = series / std(series)
            signal_fft = fft.fft(series_norm)
            freq = fft.fftfreq(len(series_norm), d=T)
            positive_freq = freq[:len(freq) // 2][1:]
            power_spectrum = (abs(signal_fft)**2) / len(series_norm)
            positive_power = power_spectrum[:len(signal_fft) // 2][1:]
            
            # Encontrar la frecuencia más cercana a la objetivo
            if len(positive_freq) > 0:
                idx_closest = np.argmin(np.abs(positive_freq - target_freq))
                powers_nori.append(positive_power[idx_closest])
            else:
                powers_nori.append(0.0)
        
        # Convertir a arrays
        powers_ri = array(powers_ri) #/ nanmax(powers_ri)
        powers_nori = array(powers_nori) #/ nanmax(powers_nori)
        
        # Filtrar valores cero o muy pequeños
        valid_ri = powers_ri[powers_ri > 1e-10]
        valid_nori = powers_nori[powers_nori > 1e-10]
        
        if len(valid_ri) < 3 or len(valid_nori) < 3:
            # No hay suficientes datos válidos para este período
            continue
        
        # Test estadístico
        try:
            _, p_value = mannwhitneyu(valid_ri, valid_nori, alternative='two-sided')
        except Exception as e:
            print(f"Error en período {target_period:.2f}h: {e}")
            continue
        
        # Calcular métricas
        median_ri = nanmedian(valid_ri)
        median_nori = nanmedian(valid_nori)
        median_diff = median_ri - median_nori
        
        # Calcular el ratio (alternativa al porcentaje)
        if median_nori > 0:
            ratio_ri_nori = (median_ri / median_nori)*100
        else:
            ratio_ri_nori = inf if median_ri > 0 else 1.0
        
        # Determinar significancia y clasificación
        is_significant = p_value < 0.05
        
        if is_significant:
            if median_diff > 0:
                classification = "RIS"
            else:
                classification = "NIS"
        else:
            classification = "NS"
        
        results_list.append({
            'period_hours': target_period,
            'period_minutes': target_period * 60,
            'frequency_hz': target_freq,
            'median_ri': median_ri,
            'median_nori': median_nori,
            'median_diff': median_diff,
            'ratio_ri_nori': ratio_ri_nori,
            'p_value': p_value,
            'significant': is_significant,
            'classification': classification,
        })
    
    return DataFrame(results_list)


def get_fourier_with_integral(series_temp, T=1/6, p0=None, p1=None):
    """
    Calcula la transformada de Fourier y opcionalmente integra la potencia 
    en un rango específico de períodos.
    
    Parameters:
    -----------
    series_temp : array-like
        Serie temporal a analizar
    T : float
        Período de muestreo en horas (default: 1/6 = 10 minutos)
    p0 : float, optional
        Período mínimo para integración (horas)
    p1 : float, optional
        Período máximo para integración (horas)
    
    Returns:
    --------
    period : array
        Períodos correspondientes a las frecuencias positivas (horas)
    positive_power : array
        Espectro de potencia para las frecuencias positivas
    integrated_power : float or None
        Integral de la potencia en el rango [p0, p1] si se especifica
    """
    # Verificar que la serie tenga datos
    if len(series_temp) == 0:
        return np.array([]), np.array([]), None
    
    # Time vector (opcional, para verificaciones)
    t = np.arange(0, len(series_temp) * T, T)
    y = series_temp
    
    # Perform Fourier Transform
    signal_fft = fft.fft(y)
    freq = fft.fftfreq(len(y), d=T)
    power_spectrum = (np.abs(signal_fft)**2) / len(y)
    
    # Remove negative frequencies and DC component
    positive_freq = freq[:len(freq) // 2][1:]
    positive_power = power_spectrum[:len(freq) // 2][1:]
    
    # Solo calcular períodos si hay frecuencias positivas
    if len(positive_freq) == 0:
        return np.array([]), np.array([]), None
    
    period = 1 / positive_freq
    
    # Diagnostic information
    #print(f"Serie length: {len(series_temp)} puntos")
    #print(f"Período de muestreo: {T:.3f} horas ({T*60:.1f} minutos)")
    #print(f"Rango de períodos disponible: {period.min():.2f} - {period.max():.2f} horas")
    
    # Initialize integrated_power to None
    integrated_power = None
    
    # Compute the integral of power between p0 and p1 if provided
    if p0 is not None and p1 is not None:
        #print(f"Rango solicitado: {p0} - {p1} horas")
        
        # Verificar que el rango solicitado esté disponible
        #if p1 > period.max() or p0 < period.min():
            #print(f"⚠️  ADVERTENCIA: Rango solicitado ({p0}-{p1}h) fuera del disponible ({period.min():.2f}-{period.max():.2f}h)")
        
        # Filtrar períodos en el rango de interés
        # CORRECCIÓN IMPORTANTE: Usar períodos directamente, no frecuencias
        period_mask = (period >= p0) & (period <= p1)
        
        if np.any(period_mask):
            # Ordenar por período para integración correcta
            sorted_indices = np.argsort(period[period_mask])
            periods_in_range = period[period_mask][sorted_indices]
            powers_in_range = positive_power[period_mask][sorted_indices]
            
            #print(f"Puntos en rango: {len(periods_in_range)}")
            
            # Integrar usando regla del trapecio
            # Como estamos integrando en función del período (no frecuencia),
            # usamos directamente período como variable de integración
            integrated_power = trapezoid(powers_in_range, periods_in_range)
            
            #print(f"Potencia integrada: {integrated_power:.6f}")
        else:
            print('❌ No hay datos en el rango especificado')
            integrated_power = 0.0
    
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
                mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)

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
#%% DATOS
df = read_csv(path_general+'datos.csv',index_col=0)
df = df[df['hours_from_max'].between(-100,0)]
df.index = df.hours_from_max
ri_labels = unique(df[df['RI']==1]['ATCF_ID'])
nori_labels = unique(df[df['RI']==0]['ATCF_ID'])
#%% SERIES SIN TENDENCIA
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


#%% ANALISIS
p0 = .1; p1 = 48
periods_analysis = (analyze_specific_periods(series_ri, series_nori)
                    .drop_duplicates(subset=['median_ri', 'median_nori','median_diff'], 
                                     keep='first'))
periods_analysis.query("period_hours>7").query("period_hours<14")#.query('classification=="RIS"')


#%% FIGURA FINAL
pows_ri_R1 = []
pows_ri_R2 = []
pows_ri_R3 = []
pows_ri_R4 = []
for i in range(len(series_ri)):
    p_temp,pow_temp,integral_R1 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=5.0,p1=6.2)
    p_temp,pow_temp,integral_R2 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=9.3,p1=10.4)
    p_temp,pow_temp,integral_R3 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=6.8,p1=7.4)
    p_temp,pow_temp,integral_R4 = get_fourier_with_integral(series_ri[i]/std(series_ri[i]),p0=7.0,p1=8.9)
    pows_ri_R1.append(integral_R1)
    pows_ri_R2.append(integral_R2)
    pows_ri_R3.append(integral_R3)
    pows_ri_R4.append(integral_R4)

pows_nori_R1 = []
pows_nori_R2 = []
pows_nori_R3 = []
pows_nori_R4 = []
for i in range(len(series_nori)):
    p_temp,pow_temp,integral_R1 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=5.0,p1=6.2)
    p_temp,pow_temp,integral_R2 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=9.3,p1=10.4)
    p_temp,pow_temp,integral_R3 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=6.8,p1=7.4)
    p_temp,pow_temp,integral_R4 = get_fourier_with_integral(series_nori[i]/std(series_nori[i]),p0=7.0,p1=8.9)
    pows_nori_R1.append(integral_R1)
    pows_nori_R2.append(integral_R2)
    pows_nori_R3.append(integral_R3)
    pows_nori_R4.append(integral_R4)
type_colors = {'NS':  (0.45, 0.45, 0.45,0.1),
               'RIS': (0.329, 0.043, 0.055, 0.7),
               'NIS': (0.263, 0.380, 0.933, 0.7)}

fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='None', edgecolor='None')

# Estilo de ejes
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color((0.45, 0.45, 0.45))
ax.tick_params(axis='both', which='major', labelsize=8.5, colors=(0.45, 0.45, 0.45))
ax.set_xlabel('Period [Hours]', color=(0.45, 0.45, 0.45), fontsize=10)
ax.set_ylabel('Difference of medians of the spectrum', color=(0.45, 0.45, 0.45), fontsize=10)
ax.axhline(y=0, color=(0.45, 0.45, 0.45), linestyle='--', alpha=0.7)

# Filtro por period_diff > 1
filtered = periods_analysis.copy()

# Graficar por tipo
for tipo, grupo in filtered.groupby("classification"):
    ax.scatter(grupo["period_hours"], grupo["median_diff"],
               s=(abs(grupo["ratio_ri_nori"] // 3) )*5 ,
               color=type_colors[tipo],
               label=tipo, edgecolors='none')

legend_elements = [
    Patch(facecolor=(0.329, 0.043, 0.055, 0.7), label='Significant Rapid Intensification'),
    Patch(facecolor=(0.263, 0.380, 0.933, 0.7), label='No Rapid Significant Intensification')
]
legend = ax.legend(handles=legend_elements, loc='upper center',ncol=2,
                   bbox_to_anchor=(0.5, 1.1), 
                   frameon=False, fontsize=8.5, title_fontsize=9)

ax.set_ylim(-.4,.4)
ax.set_xlim(0,20)
for text in legend.get_texts():
  text.set_color((0.45, 0.45, 0.45))
  text.set_fontsize(10)
    
# Insertar axes 1
inset_ax1 = inset_axes(ax, width="25%", height="35%",
                      loc='upper right',
                      bbox_to_anchor=(0, -0.04, 1, 1),
                      bbox_transform=ax.transAxes)
inset_ax1.patch.set_facecolor('none')

for spine in ['top', 'right']:
    inset_ax1.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    inset_ax1.spines[spine].set_color((0.45, 0.45, 0.45))

inset_ax1.tick_params(axis='both', which='major', labelsize=6.5,colors=(0.45, 0.45, 0.45))

histplot(x=pows_ri_R4, color="#540b0e", kde=False,ax=inset_ax1,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_ri_R4, color="#540b0e", ax=inset_ax1, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3
histplot(x=pows_nori_R4, color="#4361ee", kde=False,ax=inset_ax1,label=False,
         bins=15,edgecolor=(0.45, 0.45, 0.45),line_kws={'linestyle': '--'},stat="density")
kdeplot(data=pows_nori_R4, color="#4361ee", ax=inset_ax1, linestyle='--',
        bw_adjust=.5, common_norm=False) #adjust5,3

inset_ax1.axvline(median(pows_ri_R4),color='#540b0e')
inset_ax1.axvline(median(pows_nori_R4),color='#4361ee')
inset_ax1.set_ylabel(f'Density', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax1.set_xlabel(f'Power', color=(0.45, 0.45, 0.45), fontsize=6.5)
inset_ax1.set_title('Power Distribution Area of\nHot Towers'+ ' ' +f'over {7}-{9} Hours' , fontsize=7, y=0.99,fontweight='bold',color=(0.45, 0.45, 0.45))

ax.annotate('',
            xy=(0.74, .97), xycoords='axes fraction',
            xytext=(9, .25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))


ax.annotate('',
            xy=(0.72, .57), xycoords='axes fraction',
            xytext=(9, .25), textcoords='data',
            arrowprops=dict(arrowstyle='-', color=(0.45, 0.45, 0.45), linewidth=1, alpha=0.7))

ovalo = Ellipse((7.5, .23), 2.8, .34, fill=False, 
                linestyle=':', edgecolor=(0.45, 0.45, 0.45), linewidth=1.5, alpha=.7)
ax.add_patch(ovalo)


