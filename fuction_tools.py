"""
Codigo de funciones de la Tesis

__author__: "Duvan Nieves"
__copyright__: "UNAL"
__version__: "0.0.4"
__maintaner__:"Duvan Nieves"
__email__:"dnieves@unal.edu.co"
__status__:"Developer"
__guides__:
    - [Link catalogo HIMAWARI](https://thredds.nci.org.au/thredds/catalog/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/2015/07/01/0000/catalog.html?dataset=ra22/2015/07/01/0000/20150701000000-P1S-ABOM_OBS_B16-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc)
    - [Link sercer](https://thredds.nci.org.au/thredds/fileServer/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/2015/07/01/0000/20150701000000-P1S-ABOM_BRF_B01-PRJ_GEOS141_1000-HIMAWARI8-AHI.nc)
__refereces__:
    - [INTRO HIMAWARI](https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-himawari/2020July07_JMA_Himawari.pdf)
    - [AWS INTRO HIMAWARI](https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-himawari/2020July08_NOAA_Himawari.pdf)
    - [AWS PATH HIMAWARI](https://noaa-himawari8.s3.amazonaws.com/index.html)
    - [AWS Guide name HIMAWARI](https://noaa-himawari8.s3.amazonaws.com/README.txt)
    - [BANDAS HIMAWARI](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html#band)
    - [PATH NCI ANTIGUO HIMAWARI](https://dapds00.nci.org.au/thredds/catalog/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/catalog.html)
    - [PATH NCI NUEVO HIMAWARI](https://thredds.nci.org.au/thredds/catalog/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/catalog.html)
    - [FUENTE-CITA HIMAWARI](https://opus.nci.org.au/display/NDP/Himawari-AHI%2C+Full+Disk%2C+Brightness+Temperature)
    - [GOES](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/goes16_download.cgi)
    - [GOES_PY](https://github.com/blaylockbk/goes2go?tab=readme-ov-file)
    - [GOES PRODUCTS](https://github.com/blaylockbk/goes2go/blob/main/goes2go/product_table.txt)
    - [Listado GOES](https://inventory.ssec.wisc.edu/inventory/?date=2011/10/18&time=&satellite=GOES-16&search=1#search&start_time:2017-12-19%2000:00;end_time:2017-12-19%2023:59;satellite:GOES-16;)
    - [Listado AWS GOES](https://noaa-goes17.s3.amazonaws.com/index.html#ABI-L1b-RadF/)
    - [inicios GOES](https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf)
    - [Proyeccion GOES](https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm)
    - [Proyeccion GOES](https://lsterzinger.medium.com/add-lat-lon-coordinates-to-goes-16-goes-17-l2-data-and-plot-with-cartopy-27f07879157f)
    - [Proyeccion GOES](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php)
__methods__:
__comands__:
__changues__:
    - [2024-11-27][Duvan]: Primera version del codigo.
"""

# Modules
from pandas import date_range,DataFrame,Series,cut
from pandas import concat as pd_concat
from scipy.interpolate import CubicSpline
from datetime import timedelta
from os import system,path
from xarray import open_dataset,DataArray,Dataset,apply_ufunc,concat
from shapely.geometry import Point, Polygon, mapping,MultiPolygon
from geopandas import GeoDataFrame
from pyproj import CRS
import rioxarray as rio
from dask import config
from dask.array import from_array,stack
from gc import collect
from numpy import (array,nonzero,cov,isinf,isnan,sqrt,mean,where,linspace,
                   sin,cos,pi,zeros_like,sum,linalg,nan,deg2rad,random,log2,
                   repeat,nanmedian,fft,arange,polyval,trapz)
from numpy import round as np_round
from numpy import abs as np_abs
from scipy.ndimage import label
from scipy.stats import linregress
from time import sleep
from pymannkendall import original_test, sens_slope
from seaborn import regplot
from emd.sift import ensemble_sift
from emd.spectra import frequency_transform,hilberthuang
import re

# Metadata
## nombres
names = {
    'WP262016': 'MEARI',
    'WP062021': 'CHAMPI',
    'WP102018': 'MARIA',
    'EP112019': 'JULIETTE',
    'WP202019': 'HAGIBIS',
    'WP262019': 'FENGSHEN',
    'WP032018': 'JELAWAT',
    'WP342018': 'MAN-YI',
    'WP222019': 'BUALOI',
    'WP272017': 'SAOLA',
    'WP272015': 'IN-FA',
    'WP022016': 'NEPARTAK',
    'WP022021': 'SURIGAE',
    'WP252017': 'LAN',
    'WP132015': 'SOUDELOR',
    'WP162016': 'MERANTI',
    'WP262018': 'MANGKHUT',
    'WP142017': 'BANYAN',
    'EP202018': 'ROSA',
    'EP212018': 'SERGIO',
    'EP062022': 'ESTELLE',
    'WP152020': 'KUJIRA',
    'WP202016': 'MEGI',
    'WP232020': 'ATSANI',
    'EP022021': 'BLANCA',
    'EP052020': 'CRISTINA',
    'EP072021': 'GUILLERMO',
    'EP172022': 'PAINE',
    'WP032017': 'MUIFA',
    'WP252021': 'MALOU',
    'AL082019': 'GABRIELLE',
    'AL202021': 'VICTOR',
    'AL062018': 'FLORENCE',
    'AL122017': 'JOSE',
    'EP122019': 'AKONI',
    'AL062022': 'EARL'
}
## Paletas
paleta_cian_rojo = ['#00F0FF', '#00CCFF', '#0099FF', '#0088FF', '#0066FF', '#2000FF', '#4000FF', '#7000FF', '#9000FF', '#C000FF', '#E000FF', '#FF00CC', '#FF00AA', '#FF0080', '#FF0066', '#FF0044', '#FF0033', '#FF0022', '#FF0000']
paleta_amarillo_verde_azul = ['#FFFF00', '#FFD800', '#FFB300', '#E6C800', '#CCD600', '#99FF33', '#66FF33', '#33FF66', '#00FF99', '#00FFBB', '#00FFFF', '#00FFDD', '#00FFEE', '#00F5FF', '#00E8FF', '#00DDFF', '#00BBFF', '#00AAFF', '#0099FF']

## THRESHOLD
valores_comparacion = {
    '12': 20 }

# Fuctions
#Funcion para leer SHIPS
def read_ships(path_file):
    hours_columns = ['-12', '-6', '0', '6', '12','18', '24', '30', '36', '42', 
                     '48', '54', '60', '66', '72', '78', '84', '90', '96', 
                     '102', '108', '114', '120']
    
    final_colum = ['var','NPI','ID','date','hour','ATCF_ID']
    if path_file.split('_')[-1][0] == '5':
        init_line = 23
        columns = hours_columns + final_colum
    elif path_file.split('_')[-1][0] == '7':
        init_line = 31
        columns = hours_columns + ['126','132','138','144','150','156','162','168'] + final_colum
    with open(path_file) as file:
        #raw_ships = file.readlines()
        ships_list = []
        for line in file:
            if 'HEAD' in line:
                metadata = [re.split(r'\s+', line)[i] for i in [1,2,3,8]]
            elif ('TIME' not in line) and ('LAST' not in line):
                #clean_line = ','.join((re.split(r'\s+', line) + metadata)[1:])
                raw_clean_line = (re.split(r'\s+', line))[1:]
                if len(raw_clean_line) == init_line:
                    clean_line = ['',''] + raw_clean_line + metadata
                elif len(raw_clean_line) == init_line+1:
                    clean_line = ['',''] + raw_clean_line[:-1] + metadata
                elif len(raw_clean_line) == init_line+2:
                    clean_line = raw_clean_line + metadata
                elif len(raw_clean_line) == init_line+3:
                    clean_line = raw_clean_line[:-1] + metadata
                ships_list.append(clean_line)
        ships = DataFrame(ships_list,columns=columns)
    return ships
    
# Funcion para lectura de CPT
def loadCPT(path):
    import numpy as np                       # Import the Numpy package
    from colorsys import hsv_to_rgb

    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=np.append(x,float(ls[0]))
            r=np.append(r,float(ls[1]))
            g=np.append(g,float(ls[2]))
            b=np.append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x=np.append(x,xtemp)
        r=np.append(r,rtemp)
        g=np.append(g,gtemp)
        b=np.append(b,btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb

    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0

    xNorm = (x - x[0])/(x[-1] - x[0])

    red   = []
    blue  = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict


# Funcion para interpolar trayectorias
def interpolate_trajectory(df, hurricane_id,columns_to_interpolate, time_interval='10min'):
    """
    Interpolates hurricane trajectory data to create evenly spaced time intervals using cubic spline interpolation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing hurricane trajectory data with datetime index
    hurricane_id : str
        ATCF ID of the hurricane
    columns_to_interpolate : list
        List of column names to interpolate
    time_interval : str, default='10min'
        Desired time interval between points (pandas frequency string)
    
    Returns:
    --------
    pandas.DataFrame
        New DataFrame with interpolated values at regular time intervals
        
    Notes:
    ------
    - Uses scipy.interpolate.CubicSpline for interpolation
    - Returns NaN for columns where interpolation fails
    - Preserves original ATCF_ID in output DataFrame
    """
    df = df.sort_index()
    new_index = date_range(start=df.index.min(), 
                           end=df.index.max(), 
                           freq=time_interval)
    
    interpolated_data = {'date_time': new_index, 'ATCF_ID': hurricane_id}
    
    for column in columns_to_interpolate:
        try:
            cs = CubicSpline(df.index, df[column])
            interpolated_data[column] = cs(new_index)
        except:
            interpolated_data[column] = nan
    
    new_df = DataFrame(interpolated_data)
        
    return new_df

#Funcion para descargar y recortar datos de GOES
def process_time_goes(date,interpolated_gdf,path_raw_goes,path_cut_goes,bands,hur_sel,satellite):
    from goes2go.data import goes_timerange
    """
    Downloads and processes GOES satellite data for a specific hurricane and time period.

    Parameters:
    -----------
    date : datetime
        Target date and time for data processing
    interpolated_gdf : GeoDataFrame
        Interpolated hurricane trajectory data
    path_raw_goes : str
        Directory path for downloading raw GOES data
    path_cut_goes : str
        Directory path for saving processed GOES data
    bands : list
        List of GOES bands to download
    hur_sel : str
        Hurricane identifier
    satellite : int
        GOES satellite number (e.g., 16, 17)

    Notes:
    ------
    - Downloads GOES ABI-L2-CMIP data for specified time range
    - Clips data to region around hurricane location
    - Saves processed data in netCDF format
    - Cleans up raw downloaded files after processing
    - Uses threading for parallel processing
    - Handles errors by printing exception details
    """
    try:
        CMIP_DATA = goes_timerange(date.strftime('%Y-%m-%d %H:00'),
                                    end=(date + timedelta(hours=1)).strftime('%Y-%m-%d %H:00'),
                                    satellite=f'noaa-goes{satellite}', product='ABI-L2-CMIP', domain='F',
                                    return_as='filelist', download=True, save_dir=path_raw_goes,
                                    max_cpus=30, bands=bands)
        CMIP_DATA['file'] = path_raw_goes + CMIP_DATA['file']

        for j, row2 in CMIP_DATA.iterrows():
            if path.exists(path_cut_goes + "/OUTER/" +f"{hur_sel}/"+ row2['file'].split('/')[-1]):
                try:
                    system(f'rm {row2["file"]}')
                except:
                    pass
                continue

            with open_dataset(row2['file'],chunks='auto') as xds:
                interpolate_data = interpolated_gdf[interpolated_gdf['date_time'] == row2['creation'].round('10T')]
                gpd_point = GeoDataFrame(geometry=[Point(interpolate_data['TLON'], interpolate_data['TLAT'])], crs='epsg:4326')

                cc = CRS.from_cf(xds.goes_imager_projection.attrs)
                xds = xds.drop_dims(['number_of_image_bounds', 'number_of_time_bounds'])
                xds = xds.squeeze().transpose('y', 'x')
                xds.coords["goes_imager_projection"] = xds.goes_imager_projection
                xds.coords["t"] = xds.t
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
                subset = xds.isel(y=slice(None, None, -1))

                gpd_buffer_outer = gpd_point.to_crs(cc).buffer(1000_000)
                gpd_buffer_outer = array(gpd_buffer_outer.geometry.iloc[0].exterior.coords) / sat_height
                gpd_buffer_outer = GeoDataFrame(geometry=[Polygon(gpd_buffer_outer)], crs=cc)

                gpd_point_reproject = gpd_point.to_crs(cc)
                gpd_point_reproject = array(gpd_point_reproject.geometry.iloc[0].coords[0]) / sat_height
                gpd_point_reproject = GeoDataFrame(geometry=[Point(gpd_point_reproject)], crs=cc)

                xds_outer = subset.rio.clip(gpd_buffer_outer.geometry.apply(mapping), gpd_buffer_outer.crs)

                encoding_outer = {var: {'zlib': True, 'complevel': 4} for var in subset.data_vars}

                #for dataset in [xds_outer, xds_ring_04, data_full]:
                for dataset in [xds_outer]:
                    for var_name, var in dataset.items():
                        if 'grid_mapping' in var.attrs:
                            del var.attrs['grid_mapping']

                with config.set(scheduler='threads', num_workers=36):
                    xds_outer.chunk({'x': 500, 'y': 500}).to_netcdf(path_cut_goes + "/OUTER/" +f"{hur_sel}/"+ row2['file'].split('/')[-1], engine='h5netcdf', encoding=encoding_outer);xds_outer.close()
                    xds.close()
                    
                try:
                    system(f'rm {row2["file"]}')
                except:
                    pass


    except Exception as excp:
        print(f'Error {excp} en la fecha {date}')

    collect()

#Funcion para descargar y recortar datos de HIMAWARI
def process_time_himawari(date,interpolated_gdf,path_raw_him,path_cut_goes,bands,hur_sel,satellite):
    """
    Downloads and processes Himawari satellite data for a specific hurricane and time period.

    Parameters:
    -----------
    date : datetime
        Target date and time for data processing
    interpolated_gdf : GeoDataFrame 
        Interpolated hurricane trajectory data
    path_raw_him : str
        Directory path for downloading raw Himawari data
    path_cut_goes : str
        Directory path for saving processed data
    bands : list
        List of Himawari bands to download
    hur_sel : str
        Hurricane identifier
    satellite : int
        Himawari satellite number (e.g., 8, 9)

    Notes:
    ------
    - Downloads Himawari AHI data from NCI THREDDS server
    - Processes each requested band separately
    - Clips data to region around hurricane location
    - Saves processed data in netCDF format
    - Cleans up raw downloaded files
    - Uses threading for parallel processing
    - Handles errors by printing exception details
   """
    url_base = 'https://thredds.nci.org.au/thredds/fileServer/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/'
    try:
        url_date = url_base + f"{date.strftime('%Y/%m/%d/%H%M/%Y%m%d%H%M00-P1S-ABOM_OBS_B')}"
        name_date = date.strftime('%Y%m%d%H%M00-P1S-ABOM_OBS_B')
        for band in bands:
            url_band = url_date + f'{str(band).zfill(2)}-PRJ_GEOS141_2000-HIMAWARI{satellite}-AHI.nc'
            name_band = name_date + f'{str(band).zfill(2)}-PRJ_GEOS141_2000-HIMAWARI{satellite}-AHI.nc'
            if path.exists(path_cut_goes + "/OUTER/" +f"{hur_sel}/"+ name_band):
                continue
            system(f'wget -P {path_raw_him} "{url_band}"')
            with open_dataset(path_raw_him + name_band,chunks='auto') as xds:
                interpolate_data = interpolated_gdf[interpolated_gdf['date_time'] == date]
                gpd_point = GeoDataFrame(geometry=[Point(interpolate_data['TLON'], interpolate_data['TLAT'])], crs='epsg:4326')

                geo_transform = xds.geostationary.attrs['GeoTransform']
                geo_transform_str = ' '.join(map(str, geo_transform))
                xds.geostationary.attrs['GeoTransform'] = geo_transform_str
                cc = CRS.from_cf(xds.geostationary.attrs)
                xds = xds.squeeze().transpose('y', 'x')
                xds.coords["geostationary"] = xds.geostationary
                xds.coords["time"] = xds.time
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = 1
                subset = xds.isel(y=slice(None, None, -1))

                gpd_buffer_outer = gpd_point.to_crs(cc).buffer(1000_000)
                gpd_buffer_outer = array(gpd_buffer_outer.geometry.iloc[0].exterior.coords) / sat_height
                gpd_buffer_outer = GeoDataFrame(geometry=[Polygon(gpd_buffer_outer)], crs=cc)

                gpd_point_reproject = gpd_point.to_crs(cc)
                gpd_point_reproject = array(gpd_point_reproject.geometry.iloc[0].coords[0]) / sat_height
                gpd_point_reproject = GeoDataFrame(geometry=[Point(gpd_point_reproject)], crs=cc)

                xds_outer = subset.rio.clip(gpd_buffer_outer.geometry.apply(mapping), gpd_buffer_outer.crs)

                encoding_outer = {var: {'zlib': True, 'complevel': 4} for var in subset.data_vars}

                #for dataset in [xds_outer, xds_ring_04, data_full]:
                for dataset in [xds_outer]:
                    for var_name, var in dataset.items():
                        if 'grid_mapping' in var.attrs:
                            del var.attrs['grid_mapping']

                with config.set(scheduler='threads', num_workers=36):
                    xds_outer.chunk({'x': 500, 'y': 500}).to_netcdf(path_cut_goes + "/OUTER/" +f"{hur_sel}/"+ name_band, engine='h5netcdf', encoding=encoding_outer);xds_outer.close()
                    xds.close()  

                try:
                    system(f'rm {path_raw_him + name_band}')
                except:
                    pass

    except Exception as excp:
        print(f'Error {excp} en la fecha {date}')

    collect()

# Función para rechunking optimizado
def optimize_chunks(data, target_chunk_size='1GB'):
    """
    Optimizes chunk sizes for xarray DataArray or Dataset objects.

    Parameters:
    -----------
    data : xarray.DataArray or xarray.Dataset
        Input data to be rechunked
    target_chunk_size : str, default='1GB'  
        Target size for chunks, passed to xarray's rechunk method

    Returns:
    --------
    xarray.DataArray or xarray.Dataset
        Rechunked data object

    Raises:
    -------
    ValueError
        If input is not a DataArray or Dataset
    """
    if isinstance(data, DataArray):
        return data.chunk({'x': 'auto', 'y': 'auto'})
    elif isinstance(data, Dataset):
        return data.rechunk(chunks=target_chunk_size)
    else:
        raise ValueError("Input must be a DataArray or Dataset")
    
#Funcion para calcular distancias
def calculate_distance(x1, y1, x2, y2, mask):
    """
    Calculates Euclidean distance between two points, applying a mask.

    Parameters:
    -----------
    x1, y1 : array_like
        Coordinates of first point
    x2, y2 : array_like
        Coordinates of second point 
    mask : array_like
        Binary mask (1 for valid points, 0 for invalid)

    Returns:
    --------
    array_like
        Masked distances (NaN for invalid points)
    """
    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return where(mask == 1, distance, nan)

#Funcion para calcular la excentricidad
def calculate_eccentricity(region):
    """
    Calcula la excentricidad de una región, ignorando NaNs.
    
    :param region: Array booleano 2D representando la región
    :return: Valor de excentricidad
    """
    y, x = nonzero(region)
    if len(x) < 2 or len(y) < 2:  # Verificar si hay suficientes puntos
        return 0
    cov_s = cov(x, y)
    if isnan(cov_s).any() or isinf(cov_s).any():  # Verificar NaNs o infinitos
        return 0
    evals, _ = linalg.eig(cov_s)
    if isnan(evals).any() or isinf(evals).any():  # Verificar NaNs o infinitos
        return 0
    return sqrt(1 - (min(evals) / max(evals)))

#Funcion apra identificar Hot Towers (HT)
def identify_hot_towers(ds, var='CMI', min_bt_cold=204, min_bt_warm=211, anvil_bt=225, 
                        ot_bt_diff_cold=2, ot_bt_diff_warm=4, min_anvil_pixels=5, 
                        anvil_radius=4, min_area=4, max_area=300, max_eccentricity=0.85):
    """
    Identifies deep convective overshooting tops (hot towers) in brightness temperature data based on Bedka et al. (2010) algorithm.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing brightness temperature data
    var : str, default='CMI'
        Name of brightness temperature variable in dataset
    min_bt_cold : float, default=204
        Minimum brightness temperature threshold for coldest towers (K)
    min_bt_warm : float, default=211 
        Minimum brightness temperature threshold for warmer towers (K)
    anvil_bt : float, default=225
        Maximum brightness temperature for anvil pixels (K)
    ot_bt_diff_cold : float, default=2
        Minimum temperature difference between OT and anvil for cold towers (K)
    ot_bt_diff_warm : float, default=4
        Minimum temperature difference between OT and anvil for warm towers (K)
    min_anvil_pixels : int, default=5
        Minimum number of valid anvil pixels required
    anvil_radius : int, default=4
        Radius (in pixels) to sample anvil temperatures
    min_area : float, default=4
        Minimum area in km² for valid tower
    max_area : float, default=300
        Maximum area in km² for valid tower
    max_eccentricity : float, default=0.85
        Maximum eccentricity for circular shape consideration

    Returns:
    --------
    xarray.DataArray
        Binary mask (1 for hot towers, 0 elsewhere) with number of identified features in attributes

    Notes:
    ------
    - Implementation based on Bedka et al. (2010) methodology
    - Processes colder towers first to prioritize strongest convection
    - Assumes 2km spatial resolution for area calculations
    - Uses geometric and temperature criteria to identify overshooting tops
    - Returns zero array if no towers are identified
    """
    # Identificar píxeles fríos para ambos umbrales
    ds[var] = optimize_chunks(ds[var])
    cold_pixels_1 = ds[var] <= min_bt_cold
    cold_pixels_2 = (ds[var] > min_bt_cold) & (ds[var] <= min_bt_warm)
    
    labeled_array_1, num_features_1 = label(cold_pixels_1.values)
    labeled_array_2, num_features_2 = label(cold_pixels_2.values)
    
    def process_region(labeled_array, i, is_colder):
        region = labeled_array == i
        area = region.sum() * 4  # Suponemos resolución de 2km
        
        if min_area <= area <= max_area:
            ecc = calculate_eccentricity(region)
            if ecc < max_eccentricity:
                y_indices, x_indices = where(region)
                y, x = int(mean(y_indices)), int(mean(x_indices))
                
                anvil_temps = []
                for angle in linspace(0, 2*pi, 16, endpoint=False):
                    dy, dx = int(anvil_radius * sin(angle)), int(anvil_radius * cos(angle))
                    if 0 <= y+dy < region.shape[0] and 0 <= x+dx < region.shape[1]:
                        temp = ds[var].values[y+dy, x+dx]
                        if temp <= anvil_bt:
                            anvil_temps.append(temp)
                
                if len(anvil_temps) >= min_anvil_pixels:
                    anvil_mean_temp = mean(anvil_temps)
                    ot_temp = ds[var].values[y, x]
                    ot_bt_diff = ot_bt_diff_cold if is_colder else ot_bt_diff_warm
                    if anvil_mean_temp - ot_temp >= ot_bt_diff:
                        return region
        
        return zeros_like(region)
    
    # Procesar regiones frías primero
    regions_1 = [from_array(process_region(labeled_array_1, i, True), chunks=cold_pixels_1.chunks) 
                 for i in range(1, num_features_1 + 1)]
    
    # Luego procesar regiones menos frías
    regions_2 = [from_array(process_region(labeled_array_2, i, False), chunks=cold_pixels_2.chunks) 
                 for i in range(1, num_features_2 + 1)]
    
    all_regions = regions_1 + regions_2
    if not all_regions:
        # Si no se identificaron regiones, devolver un array de ceros
        return DataArray(zeros_like(cold_pixels_1), dims=cold_pixels_1.dims, coords=cold_pixels_1.coords)
    
    # Combinar resultados, dando prioridad a las regiones más frías
    hot_towers = stack(all_regions).sum(axis=0)
    hot_towers = (hot_towers > 0).astype(int)  # Convertir a binario (0 o 1)

    result = DataArray(hot_towers, dims=cold_pixels_1.dims, coords=cold_pixels_1.coords)
    result.attrs['num_features'] = num_features_1 + num_features_2
    
    return result

#Funcion para identificar estrucuturas convectivas (CC)
def identify_convective_features(ds, var='CMI', rain_band_bt=(210, 220), convective_bt=(180, 210), 
                                 min_length=30, min_eccentricity=0.95, max_width=50, min_area=1000):
    """
    Identifies rain bands and vigorous convective systems using brightness temperature thresholds.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing brightness temperature data
    var : str, default='CMI'
        Name of brightness temperature variable in dataset
    rain_band_bt : tuple, default=(210, 220)
        (min, max) brightness temperature thresholds for rain bands (K)
    convective_bt : tuple, default=(180, 210)
        (min, max) brightness temperature thresholds for convective systems (K)
    min_length : float, default=30
        Minimum length of rain band in km
    min_eccentricity : float, default=0.95
        Minimum eccentricity for elongated structure classification
    max_width : float, default=50
        Maximum width of rain band in km
    min_area : float, default=1000
        Minimum area for convective systems in km²

    Returns:
    --------
    xarray.DataArray
        Binary mask (1 for identified features, 0 elsewhere) with number of features in attributes

    Notes:
    ------
    - Uses brightness temperature thresholds to identify different precipitation features
    - Classifies features as either rain bands or convective systems based on geometry
    - Rain bands require minimum length, high eccentricity, and maximum width criteria
    - Convective systems require minimum area threshold
    - Assumes 2km spatial resolution for geometric calculations
    - Returns zero array if no features are identified
    """
    ds[var] = optimize_chunks(ds[var])
    rain_band_pixels = (ds[var] >= rain_band_bt[0]) & (ds[var] <= rain_band_bt[1])
    convective_pixels = (ds[var] >= convective_bt[0]) & (ds[var] <= convective_bt[1])
    
    all_features = rain_band_pixels | convective_pixels
    labeled_array, num_features = label(all_features.values)
    
    def process_region(i):
        region = labeled_array == i
        y, x = nonzero(region)
        
        # Calcular longitud y área
        length = sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2) * 4**0.5  # 2km resolution
        area = sum(region) * 4  # área en km^2
        
        # Calcular ancho (estimación aproximada)
        width = area / length if length > 0 else 0
        
        # Calcular excentricidad
        ecc = calculate_eccentricity(region)
        
        # Determinar si es banda de lluvia o sistema convectivo
        is_rain_band = (length >= min_length and 
                        ecc >= min_eccentricity and 
                        width <= max_width and
                        any(rain_band_pixels.values[region]))
        
        is_convective = (area >= min_area and
                         any(convective_pixels.values[region]))
        
        if is_rain_band or is_convective:
            return region
        return zeros_like(region)

    regions = [from_array(process_region(i), chunks=all_features.chunks) 
               for i in range(1, num_features + 1)]
        
    if not regions:
        # Si no se identificaron regiones, devolver un array de ceros
        result = DataArray(zeros_like(all_features), dims=all_features.dims, coords=all_features.coords)
        result.attrs['num_features'] = 0
    else:
        convective_features = stack(regions).sum(axis=0)
        result = DataArray(convective_features, dims=all_features.dims, coords=all_features.coords)
        result.attrs['num_features'] = num_features
    return result

#Funcion para analizar la estructura del ciclon (HT,CC)
def analyze_tc_structure(ds, var='CMI', center_point=None):
    """
    Analyzes the structure of a tropical cyclone using brightness temperature data.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing brightness temperature data with coordinates
    var : str, default='CMI'
        Name of brightness temperature variable in dataset
    center_point : GeoDataFrame
        Single-point GeoDataFrame representing storm center location

    Returns:
    --------
    xarray.Dataset
        Combined dataset containing:
        - mask_ht: Hot towers binary mask
        - mask_rb: Rain bands binary mask
        - dist_ht: Distance of each hot tower from storm center
        - dist_rb: Distance of rain bands from storm center
        - bt_ht: Brightness temperatures of hot towers
        - bt_rb: Brightness temperatures of rain bands

    Notes:
    ------
    - Identifies convective features using hot towers and rain bands detection
    - Calculates distances from storm center for each feature type
    - Extracts brightness temperatures for identified features
    - All calculations performed in parallel using dask
    """
    hot_towers = identify_hot_towers(ds, var=var).rename('mask_ht')
    rain_bands = identify_convective_features(ds, var=var).rename('mask_rb')

    dist_ht = apply_ufunc(calculate_distance,
                         hot_towers.x, hot_towers.y,
                         center_point.geometry.iloc[0].x,
                         center_point.geometry.iloc[0].y,
                         hot_towers,
                         dask='parallelized').rename('dist_ht')
    dist_rb = apply_ufunc(calculate_distance,
                         rain_bands.x, rain_bands.y,
                         center_point.geometry.iloc[0].x,
                         center_point.geometry.iloc[0].y,
                         rain_bands,
                         dask='parallelized').rename('dist_bt')
    
    bt_ht = ds[var].where(hot_towers)
    bt_rb = ds[var].where(rain_bands)

    dataset_combinado = Dataset({
        'mask_ht': hot_towers,
        'mask_rb': rain_bands,
        'dist_ht':dist_ht,
        'dist_rb':dist_rb,
        'bt_ht':bt_ht,
        'bt_rb':bt_rb
        })
            
    return dataset_combinado

#Funcion para procesar cada imagen y analizar patrones convectivos
def process_time_convective(row, delay,path_cut_goes,hur_sel,var):
    """
    Processes tropical cyclone convective structure data for a specific time.

    Parameters:
    -----------
    row : pandas.Series
        Row containing storm data including date_time, path, coordinates (TLON, TLAT)
    delay : float
        Sleep time between file reads in seconds
    path_cut_goes : str
        Base path for saving processed convective data
    hur_sel : str
        Hurricane/storm identifier (WP prefix indicates Himawari data)
    var : str
        Name of brightness temperature variable

    Notes:
    ------
    - Checks for already processed files to avoid duplication
    - Handles both GOES and Himawari satellite data
    - Projects storm center coordinates to satellite view
    - Analyzes TC structure including hot towers and rain bands
    - Saves results as compressed netCDF files
    - Uses threading for parallel processing
    - Handles errors with logging and memory cleanup
    """
    try:
        #if path.exists(path_cut_goes + "/CONVECTIVE/" +f'{hur_sel}/' + row['path'].split('/')[-1]):
            #print(f"Fecha {row['date_time']} ya procesada")
            #return
        sleep(delay)  # Delay before reading the file
        with open_dataset(row['path'],chunks='auto') as xds:
            gpd_point = GeoDataFrame(geometry=[Point(row['TLON'], row['TLAT'])],crs='epsg:4326')
            if hur_sel[0:2] == 'WP':
                cc = CRS.from_cf(xds.geostationary.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = 1
            else:
                cc = CRS.from_cf(xds.goes_imager_projection.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
            gpd_point_reproject = gpd_point.to_crs(cc)
            gpd_point_reproject = array(gpd_point_reproject.geometry.iloc[0].coords[0]) / sat_height
            gpd_point_reproject = GeoDataFrame(geometry=[Point(gpd_point_reproject)], crs=cc)
            data_full = analyze_tc_structure(xds, var=var, center_point=gpd_point_reproject)
            encoding_full = {var: {'zlib': True, 'complevel': 4} for var in data_full.data_vars}
            for dataset in [data_full]:
                for var_name, var in dataset.items():
                    if 'grid_mapping' in var.attrs:
                        del var.attrs['grid_mapping']

            with config.set(scheduler='threads', num_workers=36):
                data_full.chunk({'x': 500, 'y': 500}).to_netcdf(path_cut_goes + "/CONVECTIVE/" + f'{hur_sel}/' + row['path'].split('/')[-1], engine='h5netcdf', encoding=encoding_full);data_full.close()

    except Exception as excp:
        print(f'Error {excp} en la fecha {row["date_time"]}')
    collect()

#Funcion para crear Xarray Radial de imagens satelitales 
def create_radial_xarray(xds, center_point, satellite_crs, sat_height, max_radius=1000_000, interval=4_000):
    """
    Creates a radial analysis of satellite data around a storm center.

    Parameters:
    -----------
    xds : xarray.Dataset
        Satellite data
    center_point : GeoDataFrame
        Storm center coordinates
    satellite_crs : CRS
        Coordinate reference system of satellite data
    sat_height : float
        Satellite height (meters) for coordinate normalization
    max_radius : int, default=1000000
        Maximum analysis radius in meters
    interval : int, default=4000
        Width of each radial interval in meters

    Returns:
    --------
    xarray.Dataset or None
        Combined dataset with radial dimension if successful, None if no data found

    Notes:
    ------
    - Creates concentric rings around storm center
    - Clips satellite data to each ring
    - Handles both polygon and multipolygon geometries
    - Normalizes coordinates by satellite height
    - Returns None if no valid data found in any interval
    """
    datasets = []
    for i in range(interval, max_radius + interval, interval):
        ring = center_point.to_crs(satellite_crs).buffer(i).difference(center_point.to_crs(satellite_crs).buffer(i - interval))

        def normalize_polygon(poly):
            exterior = array(poly.exterior.coords) / sat_height
            interiors = [array(interior.coords) / sat_height for interior in poly.interiors]
            return Polygon(exterior, interiors)

        if isinstance(ring.geometry.iloc[0], Polygon):
            ring_normalized = normalize_polygon(ring.geometry.iloc[0])
        else:  # MultiPolygon
            ring_normalized = MultiPolygon([normalize_polygon(poly) for poly in ring.geometry.iloc[0]])

        ring_normalized = GeoDataFrame(geometry=[ring_normalized], crs=satellite_crs)

        try:
            xds_ring = xds.rio.clip(ring_normalized.geometry.apply(mapping), ring_normalized.crs)
            xds_ring = xds_ring.assign_coords(radial_interval=i // 1000).expand_dims("radial_interval")
            datasets.append(xds_ring)
            #print(f"Datos encontrados para el intervalo hasta {i // 1000} km")
        except Exception as e:
            print(f"No se pudieron obtener datos para el intervalo hasta {i // 1000} km: {str(e)}")

    if datasets:
        combined_xr = concat(datasets, dim="radial_interval")
        print("Xarray creado.")
        return combined_xr
    else:
        print("No se pudo crear el xarray combinado porque no hay datos en ningún intervalo.")
        return None
    collect()
    
#Funcion para procesar cada imagen y sacar sus anillos
def process_time_rings(row,hur_sel,path_cut_goes):
    """
    Processes satellite data into radial rings around tropical cyclone center.

    Parameters:
    -----------
    row : pandas.Series
        Row containing storm data including date_time, path, Longitude, Latitude
    hur_sel : str
        Hurricane/storm identifier (WP prefix indicates Himawari data)
    path_cut_goes : str
        Base path for saving processed ring data
        
    Notes:
    ------
    - Checks for existing processed files
    - Supports both GOES and Himawari satellite data
    - Creates radial analysis at 4km intervals
    - Saves compressed netCDF output
    - Uses threading for parallel processing
    - Handles errors with logging and memory cleanup
    """
    try:
        if path.exists(path_cut_goes + "/RINGS/"+f'{hur_sel}/' + row['path'].split('/')[-1]):
            return
        with open_dataset(row['path'],chunks='auto') as xds:
            gpd_point = GeoDataFrame(geometry=[Point(row['TLON'], row['TLAT'])],crs='epsg:4326')
            if hur_sel[0:2] == 'WP':
                cc = CRS.from_cf(xds.geostationary.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = 1
            else:
                cc = CRS.from_cf(xds.goes_imager_projection.attrs)
                xds.rio.write_crs(cc.to_string(), inplace=True)
                sat_height = xds.goes_imager_projection.attrs["perspective_point_height"]
            
            xds_ring_04 = create_radial_xarray(xds, gpd_point, cc, sat_height, interval=4_000)
            encoding_ring_04 = {var: {'zlib': True, 'complevel': 4} for var in xds_ring_04.data_vars}
            for dataset in [xds_ring_04]:
                for var_name, var in dataset.items():
                    if 'grid_mapping' in var.attrs:
                        del var.attrs['grid_mapping']

            with config.set(scheduler='threads', num_workers=36):
                xds_ring_04.chunk({'x': 500, 'y': 500}).to_netcdf(path_cut_goes + "/RINGS/"+f'{hur_sel}/' + row['path'].split('/')[-1], engine='h5netcdf', encoding=encoding_ring_04);xds_ring_04.close()

    except Exception as excp:
        print(f'Error {excp} en la fecha {row["date_time"]}')
    collect()


def loadCPT(path):
    """
    Load and parse a CPT (Color Palette Table) file into a color dictionary.
    
    This function reads a CPT file format commonly used in scientific visualization
    and converts it into a format compatible with matplotlib colormaps. It supports
    both RGB and HSV color models.

    Parameters
    ----------
    path : str
        Path to the CPT file.

    Returns
    -------
    Optional[Dict[str, List]]
        A dictionary containing the color data in the format:
        {'red': [[x, r, r], ...], 'green': [[x, g, g], ...], 'blue': [[x, b, b], ...]}
        where x is the normalized position (0-1) and r,g,b are color values (0-1).
        Returns None if the file cannot be opened.

    """
    from numpy import array, append
    from colorsys import hsv_to_rgb

    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = array([])
    r = array([])
    g = array([])
    b = array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=append(x,float(ls[0]))
            r=append(r,float(ls[1]))
            g=append(g,float(ls[2]))
            b=append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x=append(x,xtemp)
        r=append(r,rtemp)
        g=append(g,gtemp)
        b=append(b,btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb

    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0

    xNorm = (x - x[0])/(x[-1] - x[0])

    red   = []
    blue  = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict

def centrar_longitud(longitud):
    """
    Normalize geographic longitude values to the [-180, 180] range.

    This function takes longitude values and adjusts them to fit within the
    standard range of -180 to 180 degrees. It's useful for normalizing geographic
    coordinates when working with maps or geospatial data.

    Parameters
    ----------
    longitude : array_like
        Longitude values to normalize. Can be a single number, a list,
        or a NumPy array.

    Returns
    -------
    array_like
        Longitude values normalized to the [-180, 180] range.
    """
    from numpy import where
    return where(longitud <= 0, longitud + 180, longitud - 180)

def asignar_colores(valores, paleta):
    """
    Assign colors to numerical values using a specific color palette.

    This function maps numerical values to colors using two methods:
    a matplotlib color palette or a custom list of colors.

    Parameters
    ----------
    values : array_like
        List or array of numerical values to be assigned colors.
    palette : matplotlib.colors.LinearSegmentedColormap or list
        Can be either:
        - A matplotlib colormap (like 'plt.cm.viridis')
        - A list of colors in any valid matplotlib format
          (color names, hex codes, RGB tuples, etc.)

    Returns
    -------
    list
        List of colors in hexadecimal format (#RRGGBB).
    """
    from matplotlib import colors as mcolors
    from matplotlib.pyplot import Normalize
    if isinstance(paleta, mcolors.LinearSegmentedColormap):
        return [mcolors.to_hex(paleta(v)) for v in Normalize()(valores)]
    elif isinstance(paleta, list):
        norm = Normalize(min(valores), max(valores))
        indices = [int(norm(v) * (len(paleta) - 1)) for v in valores]
        return [paleta[i] for i in indices]
    else:
        raise TypeError("La paleta debe ser una LinearSegmentedColormap o una lista de colores")
    
def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def stats_trend(data, variable, hours_range=(-100,0)):
    """condition
    Calculate temporal evolution statistics for tropical cyclone characteristics.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Time series data containing tropical cyclone measurements.
        Must include 'hours_from_max' column and hierarchical column structure
        with variable measurements under 'mean'.
    variable : str
        Name of the variable to analyze.
    hours_range : tuple of int, optional
        Time window relative to maximum intensity (in hours) for analysis.
        Default is (-100, 0), analyzing 100 hours before maximum intensity.
        
    Returns
    -------
    pandas.Series
        Statistical metrics including:
        - Tendencia: Mann-Kendall trend direction ('increasing', 'decreasing', or 'no trend')
        - P-valor: Statistical significance of the Mann-Kendall test
        - Significativa: Boolean indicating if p-value < 0.05
        - Pendiente: Sen's slope estimator (trend magnitude)
        - Media: Mean value over the time window
        - Desv_Std: Standard deviation
        - N_Observaciones: Number of valid observations
    
    References
    ----------
    .. [1] Mann, H.B. (1945) "Non-parametric tests against trend"
    .. [2] Sen, P.K. (1968) "Estimates of the regression coefficient based on 
           Kendall's tau"
    """
    try:
        serie = data[data['hours_from_max'].between(*hours_range)][variable]['mean'].dropna()
    except:
        serie = data[data['hours_from_max'].between(*hours_range)][variable].dropna()

    
    trend = original_test(serie)
    slope = sens_slope(serie)
    
    return Series({
        'Tendencia': trend.trend,
        'P-valor': trend.p,
        'Significativa': trend.p < 0.05,
        'Pendiente': slope.slope,
        'Media': serie.mean(),
        'Desv_Std': serie.std(),
        'N_Observaciones': len(serie)
    })


def create_plots_ts(ax, data, column, color):
    ax.plot(data['hours_from_max'], data[column]['mean'], color=color, alpha=0.5)
    ax.fill_between(data['hours_from_max'],
                   data[column]['mean'] - data[column]['std'],
                   data[column]['mean'] + data[column]['std'],
                   color=color, alpha=0.1)
    
    regplot(x=data[data['hours_from_max'].between(-100, 0)]['hours_from_max'],
            y=data[data['hours_from_max'].between(-100, 0)][column]['mean'],
            ax=ax, ci=95, scatter=False,
            line_kws=dict(color=color, linestyle=(0, (3, 1, 1, 1, 1, 1))))
    

def add_stats_ts(ax, row, is_ri=True):
    case_i = 'RI' if is_ri else 'NI'
    stats = [
        ('Trend', row[case_i]['Tendencia'].title()),
        ('P-Value', row[case_i]['P-valor']),
        ('n', f"{int(row[case_i]['count'])} ({round(row[case_i]['percent'],2)} %)"), 
        ('µ', row[case_i]['Media']),
        ('σ', row[case_i]['Desv_Std']),        
    ]
    
    x_init = 0.02
    color = '#540b0e' if is_ri else '#4361ee'
    
    for j, (stat, val) in enumerate(stats):
        y_pos = 0.95 - (j*0.05)
        
        if isinstance(val, (int, float)):
            val_text = f"{val:.4f}"
        else:
            val_text = str(val)
            
        text = ax.text(x_init, y_pos, f'{stat}: ', color = (0.45, 0.45, 0.45),
                      transform=ax.transAxes, fontsize=6, weight='bold')
        
        ax.annotate(val_text, xycoords=text, xy=(1, 0),
                   verticalalignment="bottom", fontsize=6,
                   color=color, style="italic")

def add_pie_ts(ax, df_stats, var, is_ri, size=0.11):
    bbox = ax.get_position()
    
    pie_ax = ax.figure.add_axes([
        bbox.x1 - size + 0.04,
        bbox.y1 - 0.06,           
        size,                   
        size                    
    ])
    
    case_type = 'RI' if is_ri else 'NI'
    
    pie_data = df_stats[case_type].xs('percent', level=1, axis=1).loc[var]
    
    values = [
        pie_data['increasing'],
        pie_data['no trend'],
        pie_data['decreasing']
    ]
    
    if is_ri:
        colors = ['#540b0e', '#801114', '#ac171a']  
    else:
        colors = ['#4361ee', '#6b83f2', '#93a5f6']  
    
    wedges, texts = pie_ax.pie(values, 
                              colors=colors,
                              wedgeprops=dict(width=0.62),
                              startangle=90)
    
    for i, w in enumerate(wedges):
        ang = (w.theta2 - w.theta1)/2. + w.theta1
        y = sin(deg2rad(ang))
        x = cos(deg2rad(ang))
        
        symbols = ['+', '0', '-']
        
        pie_ax.text(x*0.7, y*0.7, symbols[i], 
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='white')
    
    pie_ax.axis('equal')



def apply_hht_imf(y,repeat_y=False):
    random.seed(42)
    if repeat_y:
        y = repeat(y, 3)
        max_imfs = min(int(log2(len(y))) - 1,6)
    else:
        max_imfs = min(int(log2(len(y))) - 1,6)
    imfs = ensemble_sift(y, max_imfs=max_imfs, nensembles=50, nprocesses=20, 
                        #ensemble_noise=1
                        )
    
    # Aplicamos la transformada de Hilbert a cada IMF
    IP, IF, IA = frequency_transform(imfs, 1, 'nht')
    freq = IF
    spec = IA
    period = 1/freq
    potencia = (spec**2)/imfs.shape[1]
    phase = IP
   
    return (imfs,spec, freq, period, potencia,phase)

def restructure_hht_results(full_data, variables,repeat=False):
   results = []
   for var in variables:
       for TC in full_data['ATCF_ID'].unique():
           try:
               data0 = full_data[full_data['hours_from_max'].between(-100,0)]
               data0 = data0[data0['ATCF_ID'] == TC]
               
               if len(data0) < 4:
                   continue
                   
               data_values = data0[var].interpolate(method='linear', limit_direction='both').values
               imfs, spectrum, freq, period, potencia,phase = apply_hht_imf(data_values,repeat_y=repeat)
               
               if imfs is None or imfs.size == 0:
                   continue
                   
               metrics = {}
               for i in range(imfs.shape[1]):
                   metrics[f'freq_{i+1}'] = freq[:, i]
                   metrics[f'power_{i+1}'] = potencia[:, i]  # Ensure power is added
                   metrics[f'imf_{i+1}'] = imfs[:, i]
                   metrics[f'spectrum_{i+1}'] = spectrum[:, i]
                   metrics[f'period_{i+1}'] = period[:, i]
                   metrics[f'phase_{i+1}'] = phase[:, i]
                   metrics[f'Freq_{i+1}_dominant'] = ((nanmedian(freq[:, i]))).round(2)
               if repeat:
                   step = data0['hours_from_max'].diff().dropna().unique()[-1]
                   step = 0.1666666666666666
                   new_index = linspace(data0['hours_from_max'].min(),
                                        data0['hours_from_max'].max() + step * (2*len(data0['hours_from_max'])),
                                        imfs.shape[0]+1)
                   new_index = np_round(new_index, 2)
                   #print(new_index.min(),new_index.max())
                   try:
                       df = DataFrame(metrics,index=new_index[:-1])
                   except:
                       df = DataFrame(metrics,index=new_index)
                   #df = DataFrame(metrics,index=new_index) 
                   df.index.name = "hours_from_max"
                   
               else:
                   df = DataFrame(metrics, index=data0['hours_from_max'])
               df['Case'] = 'RI' if data0['RI'].iloc[0] == 1 else 'NI' 
               df['TC'] = TC
               df['variable'] = var
               
               results.append(df)
               
           except Exception as e:
               print(f"Error processing TC {TC}, variable {var}: {str(e)}")
               continue

   if not results:
       raise ValueError("No valid results were generated")
       
   try:
       final_df = pd_concat(results)
       metric_cols = [c for c in final_df.columns if c not in ['Case', 'TC', 'variable']]
       final_df = final_df.pivot_table(
           index='hours_from_max',
           columns=['Case', 'TC', 'variable'],
           values=metric_cols
       ).reorder_levels([1,2,3,0], axis=1)#.sort_index(axis=1, level=[0,1,2,3])
       
       return final_df
   
   except Exception as e:
       print(f"Error creating final DataFrame: {str(e)}")
       return None
   

def apply_fft(y):
    # Aplicamos la transformada de Fourier sobre cada píxel (y sería la serie temporal)
    fourier = fft.fft(y) 
    freq = fft.fftfreq(len(y)) 
    periodo = 1/freq
    potencia =(np_abs(fourier)**2)/len(y)
    return (fourier, freq, periodo, potencia)
def apply_ifft(fourier):
    inversa = fft.ifft(fourier)
    return inversa

def detrend_series(y):
    """
    Remove linear trend from time series if significant trend exists.
    
    Parameters:
    -----------
    y : array-like
        Input time series
        
    Returns:
    --------
    array-like
        Detrended series if trend exists, original series otherwise
    """
    # Remove NaN values for trend calculation
    valid_idx = ~isnan(y)
    x = arange(len(y))[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) < 3:  # Need at least 3 points for meaningful trend
        return y
        
    # Check if trend is significant (p < 0.05)
    slope, intercept, r_value, p_value, std_err = linregress(x, y_valid)
    
    if p_value < 0.05:  # Significant trend
        trend = polyval([slope, intercept], arange(len(y)))
        return y - trend
    return y

def restructure_fft_results(full_data, variables,windows=48):
    """
    Perform Fourier analysis on multiple time series with detrending.
    
    Parameters:
    -----------
    full_data : pandas.DataFrame
        Input data containing time series
    variables : list
        List of variables to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Restructured DataFrame with Fourier analysis results
    """
    results = []
    
    for var in variables:
        for TC in full_data['ATCF_ID'].unique():
            try:
                # Filter data for current TC
                data0 = full_data[full_data['hours_from_max'].between(-100,0)]
                data0 = data0[data0['ATCF_ID'] == TC]
                
                if len(data0) < 4:  # Skip if insufficient data
                    continue
                    
                # Get and prepare time series
                data_values = data0[var].interpolate(method='linear',limit_direction='both')
                
                # Detrend if necessary
                #detrended_values = detrend_series(data_values)
                #detrended_values = detrended_values - detrended_values.mean()
                detrended_values = data_values - data_values.rolling(6*windows,center=True,min_periods=6).mean()
                detrended_normaliza = detrended_values/detrended_values.std()
                
                fourier, freq, period, power  = apply_fft(detrended_normaliza.values)
                
                # Create results DataFrame
                metrics = {
                    'fourier_real': fourier.real,
                    'fourier_img': fourier.imag,
                    'period': period,
                    'power': power
                }
                
                df = DataFrame(metrics)
                df['Case'] = 'RI' if data0['RI'].iloc[0] == 1 else 'NI'
                df['TC'] = TC
                df['variable'] = var
                df.index = freq
                df.index.name = 'frequency'
                
                results.append(df)
                
            except Exception as e:
                print(f"Error processing TC {TC}, variable {var}: {str(e)}")
                continue
    
    if not results:
        raise ValueError("No valid results were generated")
        
    try:
        final_df = pd_concat(results)
        metric_cols = [c for c in final_df.columns if c not in ['Case', 'TC', 'variable']]
        
        final_df = final_df.pivot_table(
            index='frequency',
            columns=['Case', 'TC', 'variable'],
            values=metric_cols
        ).reorder_levels([1,2,3,0], axis=1)
        
        return final_df
    
    except Exception as e:
        print(f"Error creating final DataFrame: {str(e)}")
        return None
    
def integrate_in_range(df, start=None, end=None, exclude_columns=None):
    """
    Integra las columnas numéricas de un DataFrame dentro de un rango específico.
    
    Parámetros:
    df: pandas.DataFrame - El DataFrame de entrada
    start: float/int - Valor inicial del rango (opcional)
    end: float/int - Valor final del rango (opcional) 
    exclude_columns: list - Lista de columnas a excluir de la integración (opcional)
    
    Retorna:
    pandas.Series con los valores integrados para cada columna, con el rango como nombre
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Determinar el rango efectivo
    start_val = start if start is not None else df.index.min()
    end_val = end if end is not None else df.index.max()
    
    # Crear la máscara para el rango
    mask = (df.index >= start_val) & (df.index <= end_val)
    
    result = {}
    
    # Integrar cada columna numérica
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col not in exclude_columns:
            x = df.index[mask].to_numpy()
            y = df[col][mask].to_numpy()
            result[col] = trapz(y, x=x)
    
    # Crear la serie con el nombre del rango
    range_name = f"{start_val}-{end_val}"
    return Series(result, name=range_name)

def integrate_sequential(df, breakpoints, exclude_columns=None):
    """
    Integra las columnas numéricas de un DataFrame para múltiples rangos secuenciales.
    
    Parámetros:
    df: pandas.DataFrame - El DataFrame de entrada
    breakpoints: list - Lista de valores que definen los límites de los rangos
    exclude_columns: list - Lista de columnas a excluir de la integración (opcional)
    
    Retorna:
    pandas.DataFrame con las integrales para cada rango como filas
    """
    results = []
    
    # Iterar sobre los rangos consecutivos
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        
        # Calcular la integral para este rango
        range_result = integrate_in_range(
            df, 
            start=start, 
            end=end, 
            exclude_columns=exclude_columns
        )
        results.append(range_result)
    
    # Concatenar todos los resultados en un DataFrame
    return pd_concat(results, axis=1)
