
"""
Codigo para calcular intensificación de datos de SHIPS y añladiendo algunos datos de IBtracks

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.2"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__refereces__:
    - [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
        - [NetCDF](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/)
        - [CSV](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/)
        - [Metadatos](https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf)
    - [SHIPS](https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/)
        - [DATA](https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/data/AL/lsdiaga_1982_2022_sat_ts_5day.txt)
        - [METADATA](https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/data/ships_predictor_file.pdf)
__comands__:
    - wget https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/data/AL/lsdiaga_1982_2022_sat_ts_5day.txt
    - wget https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/data/EP/lsdiage_1982_2022_sat_ts_5day.txt
    - wget https://rammb.cira.colostate.edu/research/tropical_cyclones/ships/data/WP/lsdiagw_1990_2021_5day.txt
    - wget https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv
__changues__:
    - [2024-07-21][Duvan]: Primera version del codigo.
"""
#%% MODULOS 
from pandas import DataFrame,to_datetime,read_csv,to_numeric,concat,IndexSlice,Categorical
from pandas import merge as pd_merge
from numpy import nan,pi
from fuction_tools import read_ships
#%% PATHS
path_data = '/home/dunievesr/Documents/UNAL/'

#%% VARIABLES DE INTERES SHIPS
var_select = [
    'VMAX', #Vientos maximos sostenidos (kt)
    'SHRD','SHRS','SHRG', #Cortante de vientos  [0] (*10 kt vs time)
    'CSST','RSST','DSST','DSTA', #Temperatura superficial del mar [1,2] (deg C*10 vs time) --> [2,3]**mejor por mas pixeles en cuenta 
    'RHLO','RHMD','RHHI','CFLX', #Humedad relativa % - menos relevancia
    'VMPI', #Maxima intensidad potencial (kt) ? averiguar ecuacion ya que contien datos de RH
    'DTL', #Distancia al punto de tierra más cercano (km) vs tiempo
    'PSLV', # Presión del centro de masa (hPa)
    'REFC', #Relative eddy momentum flux convergence  (m/sec/day, 100-600 km avg) vs time
    'G150',#Temperature perturbation at 150 hPa (deg C*10) 
    'DIVC', #Divergencia a 800 hPa
    'PENV', #Promedio de presion superficial ((hPa-1000)*10) 
    'VVAV','VMFX','VVAC', #promedio vertical de la velocidad, ponderado por densidad, mas ammplio sin vortice (m/s *100)
    'MTPW', #Total Precipitable Water (mm*10)
    'NTMX','NDTX', #Max ocean temperature in the NCODA vertical profile (deg C*10), profundidad de esa T (m) (NDMX) los babosos la tiene mal en el pdf
    'NDML', #Profundidad de la capa de mezcla (m)
    'COHC','NOHC', #Ocean heat content from the NCODA analysis (kJ/cm2) relative to the 26 C isotherm 
    'PC00','PCM1','PCM3', #Principal components and related variables from IR imagery at t=0, t=1.5B, t=3B
    'TLAT', 'TLON' # Ubicacion
    ]

#%% ESCALA DE LAS VARIABLES SHIPS
var10 =['SHRD','SHRS','SHRG','CSST','RSST','DSST','DSTA',
        'PSLV','G150','PENV','MTPW','NTMX','TLAT', 'TLON'] #variables que estan en *10
var10_wp = ['SHRD','SHRS','SHRG','CSST','RSST','PSLV','G150','PENV','MTPW','TLAT', 'TLON'] #variables que estan en *10
var100 = ['VVAV','VMFX','VVAC'] #variables que estan en *100

#%% COLUMNAS DE INETERES SHIPS (POST)
colums_select = ['0','var','ID','date','hour','ATCF_ID'] #Originales
colums_select = ['0','12','24','36','48','72','var','ID','date','hour','ATCF_ID'] #Para Pronostico 
colums_select = [str(i) for i in range (0,120+6,6)] + ['var','ID','date','hour','ATCF_ID'] #Para Pronostico final

# %% VALORES DE COMPARACION PARA DETERMINAR RI
valores_comparacion = {
    '12': 20 }

#%% DATOS IBTRACKS
url_IB = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv'
url_IB = path_data + 'ibtracs.ALL.list.v04r01.csv'
IBTrac_df = read_csv(url_IB,keep_default_na=False ,na_values=['', 'MM'])

#%% AJUSTES DATOS BESTTRACK
IBTrac_df_select = IBTrac_df[['USA_ATCF_ID','ISO_TIME','LON','LAT','USA_SSHS','USA_ROCI','USA_SEAHGT','USA_RMW','USA_EYE','USA_WIND','STORM_SPEED','IFLAG']].iloc[1:, :]
IBTrac_df_select['ISO_TIME'] = to_datetime(IBTrac_df_select['ISO_TIME'],format="%Y-%m-%d %H:%M:%S",errors='ignore')
IBTrac_df_select.index = IBTrac_df_select['ISO_TIME']
IBTrac_df_select['USA_SSHS'] = IBTrac_df_select.USA_SSHS.astype('int')
IBTrac_df_select['USA_WIND'] = to_numeric(IBTrac_df_select['USA_WIND'], errors='coerce')
IBTrac_df_select['USA_ROCI'] = to_numeric(IBTrac_df_select['USA_ROCI'], errors='coerce')
IBTrac_df_select['USA_RMW'] = to_numeric(IBTrac_df_select['USA_RMW'], errors='coerce')
IBTrac_df_select['USA_EYE'] = to_numeric(IBTrac_df_select['USA_EYE'], errors='coerce')
IBTrac_df_select['USA_ROCI'] = IBTrac_df_select['USA_ROCI'] * 1852 #Conversion a metros 
IBTrac_df_select['USA_RMW'] = IBTrac_df_select['USA_RMW'] * 1852 #Conversion a metros 
IBTrac_df_select['USA_EYE'] = IBTrac_df_select['USA_EYE'] * 1852 #Conversion a metros 
IBTrac_df_select['USA_SEAHGT'] = to_numeric(IBTrac_df_select['USA_SEAHGT'], errors='coerce')
IBTrac_df_select['USA_SEAHGT'] = IBTrac_df_select['USA_SEAHGT'] * 3.281 #Conversion a metros 
IBTrac_df_select['STORM_SPEED'] = to_numeric(IBTrac_df_select['STORM_SPEED'], errors='coerce')
IBTrac_df_select['IFLAG'] = IBTrac_df_select['IFLAG'].replace('_','',regex=True)
IBTrac_df_select['LAT'] = IBTrac_df_select.LAT.astype('float')
IBTrac_df_select['LON'] = IBTrac_df_select.LON.astype('float')
IBTrac_df_select['DUR'] =  IBTrac_df_select.groupby('USA_ATCF_ID')['ISO_TIME'].transform(lambda x: (x.max() - x.min()).total_seconds() / 3600) #horas
IBTrac_df_select['lon2'] = IBTrac_df_select['LON'].apply(lambda x: (((x + 180) % 360) - 180))
IBTrac_df_select = IBTrac_df_select[IBTrac_df_select.ISO_TIME>='1980-01-01']
IBTrac_df_select = IBTrac_df_select[IBTrac_df_select.IFLAG.str.contains('O')] #Registros no interpolados

#%% Datos SHIPS
list_data = []
for basin_file in ['lsdiaga_1982_2022_sat_ts_5day.txt',
                   'lsdiage_1982_2022_sat_ts_5day.txt',
                   'lsdiagw_1990_2021_5day.txt']:
    #% LECTURA DE DATOS SHIPS
    ships = read_ships(path_data + basin_file) 
     

    #% AJUSTE DE DATOS SHIPS PARA BUSQUEDA DE CASOS
    ships_subset = ships[ships['var'].isin(var_select)][colums_select]#.set_index('var')
    ships_subset = ships_subset.pivot(columns=['var'],index=['ID','date','hour','ATCF_ID']).astype('float')
    try:
        ships_subset.loc[:, IndexSlice[:,var10]] = ships_subset.loc[:, IndexSlice[:,var10]].div(10)
    except:
        ships_subset.loc[:, IndexSlice[:,var10_wp]] = ships_subset.loc[:, IndexSlice[:,var10_wp]].div(10)
    ships_subset.loc[:, IndexSlice[:,var100]] = ships_subset.loc[:, IndexSlice[:,var100]].div(100)
    ships_subset = ships_subset.replace([9999,999.9,99.99], nan).reset_index()
    ships_subset['date_time'] = to_datetime(ships_subset['date'] + ships_subset['hour'],format='%y%m%d%H')
    ships_subset.drop(columns=['date','hour'],inplace=True)
    ships_subset.sort_values(['ATCF_ID','date_time'],inplace=True)

    #% SELCCION DE DATOS DE PASO CERO Y UNION CON RPONOSTICOS Y OBERVACIONES PARA CALCULO DE INTENSIDAD
    ships_subset_0 = ships_subset.set_index(['ID','ATCF_ID','date_time'])['0'].reset_index()
    ships_subset_0['paso'] = (ships_subset_0.groupby('ATCF_ID').cumcount() * 6)
    ships_subset_0['BASIN'] = ships_subset_0['ATCF_ID'].str[0:2]
    for forei in range (6,120+6,6):
        ships_subset_0[f'FVMAX_{forei}'] = ships_subset[f'{forei}']['VMAX'].values
    for obsi,lag in zip(valores_comparacion.keys(),[2]):
        ships_subset_0[f'OVMAX_{obsi}'] = ships_subset_0.groupby('ATCF_ID')['VMAX'].shift(lag)
    #% CALCULO DE INTENSIDAD CORRESPONDIENTE PARA OBSERVACIONES
    result = DataFrame(index=ships_subset_0.index)
    for col in valores_comparacion:
        result[f'INT_OVMAX_{col}'] = ships_subset_0['VMAX'].sub(ships_subset_0[f'OVMAX_{col}']).gt(valores_comparacion[col])
    #ships_subset_0['INT_OVMAX_24'] = result[['INT_OVMAX_24']]
    #ships_subset_0['O_INTENSITY'] = ships_subset_0[['INT_OVMAX_24']]#.any(axis=1)
    ships_subset_0['INT_OVMAX_12'] = result[['INT_OVMAX_12']]
    ships_subset_0['O_INTENSITY'] = ships_subset_0[['INT_OVMAX_12']]#.any(axis=1)
    # ASIGNACION DE CLASIFICACION
    ships_subset_0.loc[ships_subset_0['O_INTENSITY']>0, "O_INTENSITY"] = "RI" 
    ships_subset_0.loc[ships_subset_0['O_INTENSITY']==0, "O_INTENSITY"] = "NI" 
    #ships_subset_0['FINAL_O_INTENSITY'] = ships_subset_0.groupby(['ATCF_ID'])['INT_OVMAX_24'].cumsum()
    ships_subset_0['FINAL_O_INTENSITY'] = ships_subset_0.groupby(['ATCF_ID'])['INT_OVMAX_12'].cumsum()
    ships_subset_0['FINAL_O_INTENSITY'] = ships_subset_0.groupby(['ATCF_ID'])['FINAL_O_INTENSITY'].transform('max')
    ships_subset_0.loc[ships_subset_0['FINAL_O_INTENSITY']>0, "FINAL_O_INTENSITY"] = "RI" 
    ships_subset_0.loc[ships_subset_0['FINAL_O_INTENSITY']==0, "FINAL_O_INTENSITY"] = "NI" 

    #% SELCCION DE RANGO DE TIEMPO DE IMAENES SATELITALES
    fecha = '2017-02-28' if 'lsdiaga' in basin_file else '2018-08-28' if 'lsdiage' in basin_file else '2015-07-03' if 'lsdiagw' in basin_file else None
    
    #ships_select=ships_select.set_index('date_time').sort_index().loc['2017-02-28':'2023'] #G16 AL,EP
    ships_select=ships_subset_0.set_index('date_time').sort_index().loc[fecha:'2023'] #HI8 WP

    # union datos ships ibtracks
    Ships_IBTracs = pd_merge(ships_select.reset_index(),
                            IBTrac_df_select.reset_index(drop=True), 
                            right_on=['USA_ATCF_ID', 'ISO_TIME'], 
                            left_on=['ATCF_ID', 'date_time'], how='left')
    Ships_IBTracs.set_index('date_time',inplace=True)
    list_data.append(Ships_IBTracs)

ships_all = concat(list_data).sort_index()
ships_all['date_time_2'] = ships_all.index
ships_all.sort_values(['ATCF_ID','date_time_2'],inplace=True)

#%% CONDICIONES ANTES DE INTENSIFICACION MAXIMA
ships_intensity_time = ships_all.copy()
#MAXIMA INTENSIDAD
ships_intensity_time['MAX_VMAX'] = ships_intensity_time[['ATCF_ID','VMAX']].groupby('ATCF_ID').transform('max')
# INDICE DONDE SE CUMPLE LA MAXIMA INTENSIDAD
ships_intensity_time.loc[ships_intensity_time['VMAX'] == ships_intensity_time['MAX_VMAX'], "MAX_INT"] = 1 
#FECHAS EN LAS QUE SE DA LA MAXIMA INTENSIDAD
ships_intensity_time['date_max'] = ships_intensity_time.apply(lambda row: row.name if row['MAX_INT'] == 1 else None, axis=1) 
#PRIMERA FECHA EN LA QUE SE CUMPLE LA INTENSIDAD MAXIMA
ships_intensity_time['date_max_min'] = ships_intensity_time[['ATCF_ID','date_max']].groupby('ATCF_ID').transform('min') 
#PASOS DE TIEMPO DONDE SE CUMPLEN LAS CONDICIONES
ships_intensity_time.loc[ships_intensity_time.index <= ships_intensity_time['date_max_min'], "IDX_DATE_MAX"] = 1 
ships_intensity_time = ships_intensity_time[ships_intensity_time['IDX_DATE_MAX']==1]
#TIEMPO DE CUMPLIMIENTO DE CONDCIONES DE INTENCIFICACION

#%% CONDICIONES AMBIENTALES
ships_environmental_conditions = (
    ships_intensity_time[['BASIN', 'FINAL_O_INTENSITY', 'ATCF_ID','VMAX', 'VMPI', 'SHRD', 'RSST']]
    .assign(
        favorable_vmpi = lambda x: (x.groupby('ATCF_ID')['VMPI']
                                  .transform(lambda y: ((y >= 140) * 1)
                                                     .groupby((y >= 140).ne(1).cumsum())
                                                     .cumsum())),
        
        low_shear = lambda x: (x.groupby('ATCF_ID')['SHRD']
                              .transform(lambda y: ((y <= 15) * 1)
                                                 .groupby((y <= 15).ne(1).cumsum())
                                                 .cumsum())),
        
        warm_sst = lambda x: (x.groupby('ATCF_ID')['RSST']
                             .transform(lambda y: ((y >= 28.5) * 1)
                                                .groupby((y >= 28.5).ne(1).cumsum())
                                                .cumsum()))
    )
)
ships_environmental_conditions['favorable_conditions'] = (
    ships_environmental_conditions[['favorable_vmpi','low_shear','warm_sst']]
    .gt(0).all(axis=1)  # Verifica todas las condiciones favorables
    .mul(1)             # Convierte a 1s y 0s
    .groupby(          # Agrupa por ATCF_ID y por períodos donde la condición cambia
        [ships_environmental_conditions['ATCF_ID'],
         ships_environmental_conditions[['favorable_vmpi','low_shear','warm_sst']]
         .gt(0).all(axis=1).ne(1).cumsum()]
    )
    .cumsum()          # Suma acumulativa que se reinicia
)
ships_environmental_conditions['favorable_rate'] = ships_environmental_conditions.groupby('ATCF_ID')['favorable_conditions'].diff() 


#%% CASOS SELCCIONADOS
codes_select = [
    "WP262016", "WP062021", "WP102018", "EP112019", "WP202019", "WP262019",
    "WP032018", "WP342018", "WP222019", "WP272017", "WP272015", "WP022016",
    "WP022021", "WP252017", "WP132015", "WP162016", "WP262018", "WP142017",
    "EP202018", "EP212018", "EP062022", "WP152020", "WP202016", "WP232020",
    "EP022021", "EP052020", "EP072021", "EP172022", "WP032017", "WP252021",
    "AL082019", "AL202021", "AL062018", "AL122017", "EP122019", "AL062022"
]


#%% DATOS DE RANGO TEMPORAL
final_select = ships_environmental_conditions[ships_environmental_conditions['ATCF_ID'].isin(codes_select)]
range_data = (final_select
 .reset_index()
 .groupby('ATCF_ID')
 .agg(MI_DATE = ('date_time','min'),
      MA_DATE = ('date_time','max'),
      VMAX=('VMAX','max'),
      SST=('RSST','max'),
      TIME=('ATCF_ID',lambda x: x.count()*6),
      MNCDPC = ('favorable_conditions','max'),
      NSBMI = ('favorable_conditions','count'),
      INT = ('FINAL_O_INTENSITY','unique')))
range_data.to_csv('range_data12.csv')

#%% DATOS PARA CASOS SELECCIONADOS
traj = ships_all[ships_all['ATCF_ID'].isin(codes_select)] 
traj = traj[['ATCF_ID','TLAT','TLON','VMAX','RSST','VMPI','RHLO','SHRD','PSLV','NOHC','STORM_SPEED','USA_ROCI','USA_RMW','USA_EYE','FINAL_O_INTENSITY']]
traj = pd_merge(traj.reset_index(),range_data[['MNCDPC']].reset_index(),
        right_on=['ATCF_ID'],
        left_on=['ATCF_ID',], how='left')
traj.sort_values(by=['ATCF_ID','date_time']).to_csv('trayectories12.csv',index=False)
#%%
