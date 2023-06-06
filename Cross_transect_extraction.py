#Code to extract transects

## SET Working directory
import os
os.chdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Codes')

#%% Import packages needed

from matplotlib import pyplot as plt
from pandas import ExcelWriter
from geoimread import geoimread
#import rasterio as rs 
#import rasterio.plot
import pandas as pd
import numpy as np
#from scipy.interpolate import interp1d
#from pyproj import Proj
#from scipy.interpolate import griddata
#from scipy.interpolate import RectBivariateSpline
#from scipy import signal
#from osgeo import gdal
#from osgeo import gdalconst

#%%Define excel save 
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()

#%% Define path for importing filtered displacement maps
xls_path_df_net_disp = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Triangulation_input/df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked.xlsx'

#Clean first column that gets created due to saving in the block above
df_net_disp_imported = pd.read_excel(xls_path_df_net_disp, sheet_name=None)
for i in range(33):
    del df_net_disp_imported[list(df_net_disp_imported.keys())[i]]["Unnamed: 0"]    
    
keys = list(df_net_disp_imported.keys())

df_net_disp_imported_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_net_disp_imported[keys[i]]) 
    df_net_disp_imported_list.append(df)
    del df

del df_net_disp_imported
del keys

#%% Convert displacements to velocities using time differences

# Create string list for each pair
all_img_tiffile = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all')

datestring_all_img = []
datestring_single_img =[]

for i in range(34):
    datestring_single_img = all_img_tiffile[i][ 14: 22]
    datestring_all_img.append(datestring_single_img)

from datetime import datetime
datetime_all_img = []
img_datestring = []
datetime_single_img = []

#create list with datetime format days of each image
for i in range(34):
    img_datestring = datestring_all_img[i]
    datetime_single_img  = datetime.strptime(img_datestring, '%Y%m%d')
    datetime_all_img.append(datetime_single_img)
    del img_datestring
    del datetime_single_img

timestep_img1img2 = []
timestep_days_singlepair = []
timestep_days_allpairs= []

#compute difference in days between each image
for i in range(33):  
    timestep_img1img2 = datetime_all_img[i+1] -  datetime_all_img[i]
    timestep_days_singlepair = timestep_img1img2.days
    timestep_days_allpairs.append(timestep_days_singlepair)
    del timestep_img1img2
    del timestep_days_singlepair
    
#convert displacement in velocities
Velocity_singlepair_imported = []
Velocity_allpairs_imported = []

for i in range(33):
    Velocity_singlepair_imported = df_net_disp_imported_list[i] / timestep_days_allpairs[i]
    Velocity_allpairs_imported.append(Velocity_singlepair_imported)
    del Velocity_singlepair_imported
    
# insert fixed 2019 07 04 - 2019 07 09 manually corrected
Manual_corrected_velocity_20190704_20190709 = np.load('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/ManualFT_20190704_20190709_SW21_Velocity_350x350.npy')
Manual_corrected_velocity_20190704_20190709 = pd.DataFrame(Manual_corrected_velocity_20190704_20190709)
Velocity_allpairs_imported[17] = Manual_corrected_velocity_20190704_20190709

#%% Import sampling points
xls_path_transect_Kronebreen = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Shapefiles/Kronebreen_cross_transect_points.csv'
xls_path_transect_Kongsbreen = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Shapefiles/Kongsbreen_cross_transect_points.csv'

cross_transect_Kronebreen = pd.read_csv(xls_path_transect_Kronebreen)
cross_transect_Kongsbreen = pd.read_csv(xls_path_transect_Kongsbreen)

#%% Create 350x350 of lat lon

fA = 'C:/Users/39333/OneDrive - Universitetet i Oslo\Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/S2B08_cropped_20190331.tif'
A = geoimread(fA, roi_x= None, roi_y= None, roi_crs={'init': 'EPSG:32633'}, buffer= None)

# create longitude dataframe
lon = pd.DataFrame(A.x)
lon = np.tile(lon,(1,len(lon) - 1))
lon = pd.DataFrame(lon)
lon.drop([1750],axis=0,inplace=True)

#create latitude dataframe
lat = pd.DataFrame(A.y)
lat = np.tile(lat,(1,len(lat) - 1))
lat = pd.DataFrame(lat)
lat.drop([1750],axis=0,inplace=True)

#convert to numpy in order to flatten
lat_1d = lat.iloc[:, [0]]
lon_1d = lon.iloc[:, [0]]


LatLon_df = np.column_stack((lat_1d, lon_1d))
#change is pandas dataframe
LatLon_df = pd.DataFrame(LatLon_df)
#name columns
LatLon_df.columns = ['Lat(m)', 'Lon(m)']

Disp_resolution = 50
xRange = np.arange(LatLon_df["Lon(m)"].min(),LatLon_df["Lon(m)"].max(),Disp_resolution)
yRange = np.arange(LatLon_df["Lat(m)"].min(),LatLon_df["Lat(m)"].max(),Disp_resolution)

#create grid of latitue and longitude 350x350 = 50m resolution
gridX,gridY = np.meshgrid(xRange,yRange)

gridX = pd.DataFrame(gridX)
gridY = pd.DataFrame(gridY)

#%% Create Lat Lon list of sampling cross transect points on Kronebreen
lon_index_list_cross_Kronebreen = []
lat_index_list_cross_Kronebreen = []

for i in range (cross_transect_Kronebreen.shape[0]):
    diff_grid_lon = (gridX-cross_transect_Kronebreen['X'][i].astype(int)).abs()          
    lon_index = diff_grid_lon.to_numpy().argmin()
    lon_index_list_cross_Kronebreen.append(lon_index)

    diff_grid_lat = (gridY-cross_transect_Kronebreen['Y'][i].astype(int)).abs()             
    lat_index = diff_grid_lat[0].to_numpy().argmin()
    lat_index_list_cross_Kronebreen.append(lat_index)
    
    del diff_grid_lon
    del diff_grid_lat
    del lon_index
    del lat_index
    
#%% Create Lat Lon list of sampling transect points on Kongsbreen
lon_index_list_cross_Kongsbreen = []
lat_index_list_cross_Kongsbreen = []

for i in range (cross_transect_Kongsbreen.shape[0]):
    diff_grid_lon = (gridX-cross_transect_Kongsbreen['X'][i].astype(int)).abs()          
    lon_index = diff_grid_lon.to_numpy().argmin()
    lon_index_list_cross_Kongsbreen.append(lon_index)

    diff_grid_lat = (gridY-cross_transect_Kongsbreen['Y'][i].astype(int)).abs()          
    lat_index = diff_grid_lat[0].to_numpy().argmin()
    lat_index_list_cross_Kongsbreen.append(lat_index)
        
    del diff_grid_lon
    del diff_grid_lat
    del lon_index
    del lat_index
    
#%%The position of latitude points has to be mirrored in the relative 350x350 in order to match the coordinates
#of the velocity field

#compute difference between mirror plane (half of 350 = 175) and coordinates
diff_lat_for_mirroring_cross_Kronebreen = (175 - np.array(lat_index_list_cross_Kronebreen)) * 2
diff_lat_for_mirroring_cross_Kongsbreen = (175 - np.array(lat_index_list_cross_Kongsbreen)) * 2

#sum difference to sampling points for Kronebreen
lat_index_list_cross_Kronebreen_mirrored = np.array(lat_index_list_cross_Kronebreen) + diff_lat_for_mirroring_cross_Kronebreen
lat_index_list_cross_Kronebreen_mirrored = lat_index_list_cross_Kronebreen_mirrored.tolist()

#sum difference to sampling points for Kongsbreen
lat_index_list_cross_Kongsbreen_mirrored = np.array(lat_index_list_cross_Kongsbreen) + diff_lat_for_mirroring_cross_Kongsbreen
lat_index_list_cross_Kongsbreen_mirrored = lat_index_list_cross_Kongsbreen_mirrored.tolist()


#%% Create zipped list lon (X) - lat (Y) of all the transect points
Transect_points_cross_Kronebreen = list(zip(lon_index_list_cross_Kronebreen, lat_index_list_cross_Kronebreen_mirrored))
Transect_points_cross_Kongsbreen = list(zip(lon_index_list_cross_Kongsbreen, lat_index_list_cross_Kongsbreen_mirrored))

#%%get all cross transect velocities for Kronebreen
single_value =[]
CrossTransect_Kronebreen_allvalues = []

#create full list to fill 
for x in range(33):
    CrossTransect_Kronebreen_allvalues.append([[]])

#fill list, first n is for each velocity field, second n is for each transect sampling point
for n in range (33): 
    for i in range (cross_transect_Kronebreen.shape[0]):
        single_value = Velocity_allpairs_imported[n].iat[Transect_points_cross_Kronebreen[i][1], Transect_points_cross_Kronebreen[i][0]]
        CrossTransect_Kronebreen_allvalues[n].append(single_value)
        del single_value

for i in range (33):
    CrossTransect_Kronebreen_allvalues[i].remove([])


#%%get all cross transect velocities for Kongsbreen
single_value =[]
CrossTransect_Kongsbreen_allvalues = []

#create full list to fill 
for x in range(33):
    CrossTransect_Kongsbreen_allvalues.append([[]])

#fill list, first n is for each velocity field, second n is for each transect sampling point
for n in range (33): 
    for i in range (cross_transect_Kongsbreen.shape[0]):
        single_value = Velocity_allpairs_imported[n].iat[Transect_points_cross_Kongsbreen[i][1], Transect_points_cross_Kongsbreen[i][0]]
        CrossTransect_Kongsbreen_allvalues[n].append(single_value)
        del single_value

for i in range (33):
    CrossTransect_Kongsbreen_allvalues[i].remove([])
    
#%%final colorplot 3D definition Kronebreen

df_cross_transect_Kronebreen = pd.DataFrame()

for i in range (len(CrossTransect_Kronebreen_allvalues)):
    col = np.array(CrossTransect_Kronebreen_allvalues[i])
    df_cross_transect_Kronebreen[i] = col

Distance_horizontal_Kronebreen = cross_transect_Kronebreen['distance']
Days = datestring_all_img[0:33]

#%%final colorplot 3D definition Kongsbreen

df_cross_transect_Kongsbreen = pd.DataFrame()

for i in range (len(CrossTransect_Kongsbreen_allvalues)):
    col = np.array(CrossTransect_Kongsbreen_allvalues[i])
    df_cross_transect_Kongsbreen[i] = col

Distance_horizontal_Kongsbreen = cross_transect_Kongsbreen['distance']
Days = datestring_all_img[0:33]

#%% Set LATEX font
#------------------------------
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams.update({'font.size': 8})
from datetime import datetime
#-----------------------------

#%%Plot cross transect velocities Kronebreen
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 33))

fig=plt.figure(figsize=(6, 4))                   

for i in range(33):
    plt.plot(cross_transect_Kronebreen['distance']/1000,  CrossTransect_Kronebreen_allvalues[i] , c=color[i])

plt.ylabel("Surface velocity [md$^{-1}$]", fontsize = 12)
plt.xlabel("Distance along transect [km]", fontsize= 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.ylim(top=7)
plt.ylim(bottom=0)
plt.legend(Days, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 9.5, ncol = 2)
plt.tight_layout()
plt.show()

#%%
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Transects/CrossTransect_Kronebreen_nofilters_v3.png', dpi = 500, bbox_inches='tight' )

#%%Plot cross transect velocities Kronebreen in 2019
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 11))

CrossTransect_Kronebreen_allvalues_2019 = CrossTransect_Kronebreen_allvalues[10:21]
Days_2019 = Days[10:21]

fig=plt.figure(figsize=(6, 4))                   

for i in range(10):
    plt.plot(cross_transect_Kronebreen['distance']/1000,  CrossTransect_Kronebreen_allvalues_2019[i] , c=color[i])

plt.ylabel("Surface velocity [md$^{-1}$]", fontsize = 12)
plt.xlabel("Distance along transect [km]", fontsize= 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.ylim(top=7)
plt.ylim(bottom=0)
plt.legend(Days_2019, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 9.5)
plt.tight_layout()
plt.show()

#%%
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Transects/CrossTransect_Kronebreen_2019_v2.png', dpi = 400 )

#%%Plot cross transect velocities Kongsbreen
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 33))

fig=plt.figure(figsize=(6, 4))                   

for i in range(33):
    plt.plot(cross_transect_Kongsbreen['distance']/1000,  np.flip(CrossTransect_Kongsbreen_allvalues[i]) , c=color[i])

plt.ylabel("Surface velocity [md$^{-1}$]", fontsize = 12)
plt.xlabel("Distance along transect [km]", fontsize= 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.ylim(top=7)
plt.ylim(bottom=0)
plt.legend(Days, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 9.5, ncol = 2)
plt.tight_layout()
plt.show()

#%%
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Transects/CrossTransect_Kongsbreen_nofilters_v2.png', dpi = 400 )

#%%Plot cross transect velocities Kongsbreen in 2019
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 11))

CrossTransect_Kongsbreen_allvalues_2019 = CrossTransect_Kongsbreen_allvalues[10:21]
Days_2019 = Days[10:21]

fig=plt.figure(figsize=(6, 4))                   

for i in range(10):
    plt.plot(cross_transect_Kongsbreen['distance']/1000,  np.flip(CrossTransect_Kongsbreen_allvalues_2019[i]) , c=color[i])

plt.ylabel("Surface velocity [md$^{-1}$]", fontsize = 12)
plt.xlabel("Distance along transect [km]", fontsize= 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.ylim(top=7)
plt.ylim(bottom=0)
plt.legend(Days_2019, loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 9.5)
plt.tight_layout()
plt.show()

#%%
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Transects/CrossTransect_Kongsbreen_2019.png', dpi = 400 )

