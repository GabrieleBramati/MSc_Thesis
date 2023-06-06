# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:44:25 2022
Plotting code for Sentinel 2 feature tracking - single pairs (1+1) processing and plot
@author: Gabriele Bramati
"""
import os
os.chdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Codes')

S2L1C_dir_list = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/')
#%% Import packages needed
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show 
import rasterio as rs 
import rasterio.plot
import pandas as pd
import numpy as np
from scipy import signal
#from scipy.interpolate import interp1d
#from pyproj import Proj
from scipy.interpolate import griddata
#from scipy.interpolate import RectBivariateSpline
#from osgeo import gdal
#from osgeo import gdalconst

#%% Import all tables output from feature tracking

#Define path to files from block above
xls_path_pu = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_pu_allpairs.xlsx'
xls_path_pv = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_pv_allpairs.xlsx'
xls_path_du = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_du_allpairs.xlsx'
xls_path_dv = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_dv_allpairs.xlsx'
xls_path_meanAbsCorr = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_meanAbs_Corr_allpairs.xlsx'
xls_path_peakCorr = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_peakCorr_allpairs.xlsx'
xls_path_snr = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/df_snr_allpairs.xlsx'


#Clean first column that gets created due to saving in the block above
df_pu_imported = pd.read_excel(xls_path_pu, sheet_name=None)
for i in range(33):
    del df_pu_imported[list(df_pu_imported.keys())[i]]["Unnamed: 0"]

del xls_path_pu

df_pv_imported = pd.read_excel(xls_path_pv, sheet_name=None)
for i in range(33):
    del df_pv_imported[list(df_pv_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_pv
    
df_du_imported = pd.read_excel(xls_path_du, sheet_name=None)
for i in range(33):
    del df_du_imported[list(df_du_imported.keys())[i]]["Unnamed: 0"]

del xls_path_du
    
df_dv_imported = pd.read_excel(xls_path_dv, sheet_name=None)
for i in range(33):
    del df_dv_imported[list(df_dv_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_dv

df_meanAbsCorr_imported = pd.read_excel(xls_path_meanAbsCorr, sheet_name=None)
for i in range(33):
    del df_meanAbsCorr_imported[list(df_meanAbsCorr_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_meanAbsCorr

df_peakCorr_imported = pd.read_excel(xls_path_peakCorr, sheet_name=None)
for i in range(33):
    del df_peakCorr_imported[list(df_peakCorr_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_peakCorr
    
df_snr_imported = pd.read_excel(xls_path_snr, sheet_name=None)
for i in range(33):
    del df_snr_imported[list(df_snr_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_snr
    
#%% #Convert imported dictionaries in list of dataframes
#get list of keys
keys = list(df_pu_imported.keys())

df_pu_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_pu_imported[keys[i]]) 
    df_pu_list.append(df)
    del df

del df_pu_imported
    
df_pv_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_pv_imported[keys[i]]) 
    df_pv_list.append(df)
    del df

del df_pv_imported

df_du_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_du_imported[keys[i]]) 
    df_du_list.append(df)
    del df

del df_du_imported

df_dv_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_dv_imported[keys[i]]) 
    df_dv_list.append(df)
    del df

del df_dv_imported

df_meanAbsCorr_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_meanAbsCorr_imported[keys[i]]) 
    df_meanAbsCorr_list.append(df)
    del df

del df_meanAbsCorr_imported
    
df_peakCorr_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_peakCorr_imported[keys[i]]) 
    df_peakCorr_list.append(df)
    del df

del df_peakCorr_imported
    
df_snr_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_snr_imported[keys[i]]) 
    df_snr_list.append(df)
    del df

del df_snr_imported

#%% Compute displacement for each pair
net_disp_pixels = []
net_disp_meters = []
net_disp_meters_allpairs = []

for i in range(33): #images are 34, therefore pairs are 33
    net_disp_pixels = np.sqrt(df_du_list[i] ** 2 + df_dv_list[i] ** 2)
    net_disp_meters = net_disp_pixels.multiply(10)
    net_disp_meters_allpairs.append(net_disp_meters)
    del net_disp_pixels
    del net_disp_meters



#%% Save dictionary with all displacement dataframes masked
# =============================================================================
# from pandas import ExcelWriter
# def save_xls(list_dfs, xls_path):
#     with ExcelWriter(xls_path) as writer:
#         for n, df in enumerate(list_dfs):
#             df.to_excel(writer,'sheet%s' % n)
#         writer.save()
# 
# xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw/df_net_disp_meters_allpairs_imported_masked.xlsx'
# save_xls(df_net_disp_meters_allpairs_imported_masked, xls_path)
# del xls_path
# =============================================================================

#%% Filter all displacement raster for Max correlation and apply median filter with a 3x3 kernel size
df_net_disp_meters_allpairs_imported_peakCorr_med_filtered = []
single_pair_peakCorr_filtered = []
single_pair_peakCorr_filtered_medfilt = []

for i in range(33):
#Define peakCorr threshold
    peakCorr_index = df_peakCorr_list[i].loc[:,:] > 50
#Convert "False" to nan
    peakCorr_index = peakCorr_index * 1
    peakCorr_index[peakCorr_index==0] = np.nan
#Multiply the displacement raster to where the peakCorr threshold is satisfied
    single_pair_peakCorr_filtered = net_disp_meters_allpairs[i] * peakCorr_index
#Filtering with 3x3 medial filter    
    single_pair_peakCorr_filtered_medfilt = signal.medfilt2d(single_pair_peakCorr_filtered, kernel_size=3)
    single_pair_peakCorr_filtered_medfilt = pd.DataFrame(single_pair_peakCorr_filtered_medfilt)
#Final dataframe with snr and 3x3 median filter    
    df_net_disp_meters_allpairs_imported_peakCorr_med_filtered.append(single_pair_peakCorr_filtered_medfilt)
    del peakCorr_index
    del single_pair_peakCorr_filtered
    del single_pair_peakCorr_filtered_medfilt
#%%
from pandas import ExcelWriter
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
         for n, df in enumerate(list_dfs):
             df.to_excel(writer,'sheet%s' % n)
         writer.save()
         
#%% Mask tiff for glacier from .csv to same resolution as feature tracking output
#import mask tiff
Mask_glaciers_df = pd.read_csv('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Shapefiles/KroBreKonBre_mask_glaciers_cropDisplacement_cleaned.csv')

points_glaciermask = list(zip(Mask_glaciers_df["Easting[m]"].to_list(),Mask_glaciers_df["Northing[m]"].to_list()))
values_glaciermask = Mask_glaciers_df["Mask"].to_list()

Disp_resolution = 47.9 #end up being 350x350 resample the mask on the same rasters resolution
xRange_mask = np.arange(Mask_glaciers_df["Easting[m]"].min(),Mask_glaciers_df["Easting[m]"].max(),Disp_resolution)
yRange_mask = np.arange(Mask_glaciers_df["Northing[m]"].min(),Mask_glaciers_df["Northing[m]"].max(),Disp_resolution)
yRange_mask  = np.delete(yRange_mask, -1)

gridX_mask,gridY_mask = np.meshgrid(xRange_mask,yRange_mask )

#Raster of the glacier mask with 1 (glacier) and 0 (non glacier) with the same resolution of FT output
grid_glaciermask = griddata(points_glaciermask, values_glaciermask, (gridX_mask ,np.flip(gridY_mask) ), method='nearest')

#%% Create masked displacement maps of glaciers
grid_glaciermask = grid_glaciermask.astype(np.float64)
grid_glaciermask[grid_glaciermask==0] = np.nan

df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked = []
df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked = []

#create dictionary with each FT output masked 
for i in range (33):
    df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked = df_net_disp_meters_allpairs_imported_peakCorr_med_filtered[i] * grid_glaciermask
    df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked.append(df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked)
    del df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked
#%%
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Triangulation_input/df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked.xlsx'
save_xls(df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked, xls_path)
del xls_path

#%% Perform NaN count 
subset_singlepair = []
subset_singlepair_01 = []
nan_stack = []

for i in range (33):
    subset_singlepair = df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked[i]
    #transform all the positive values into 0
    subset_singlepair[subset_singlepair>0] = 0
    #transform all the nan values into 1
    subset_singlepair_01 = subset_singlepair.fillna(1)
    nan_stack.append(subset_singlepair_01)

#sum all the nan maps
nan_stack_sum = pd.DataFrame(np.zeros((350, 350)))

for i in range(33):
    subset = nan_stack[i]
    nan_stack_sum = nan_stack_sum + subset 
    del subset

#mask the sea area
nan_stack_sum_masked = nan_stack_sum * grid_glaciermask


#%%Plot NaN graph

plt.rcParams["figure.figsize"] = [8, 6]
plt.xlabel("Easting [m]", fontsize = 10)
plt.xticks(fontsize = 8)
plt.ylabel("Northing [m]", fontsize = 10)
plt.yticks(fontsize = 8)
plt.rc('font', **{'size':'8'})

#plot nan sum values masked
ax = plt.pcolormesh(gridX_mask,np.flip(gridY_mask),nan_stack_sum_masked, alpha= 1, vmin = 0, vmax = 33)
#define colorbar
cbar = plt.colorbar(ax, extend = 'both', ticks=[0, 5, 10, 15, 20, 25, 30, 33])
cbar.set_label('NaN count', rotation=270, labelpad=15, fontsize = 10)

#%%
#Save fig
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/NaN_count/FT_NaN_count.png', dpi=400, bbox_inches='tight')
# Close open figures
plt.close('all')


#%%Create stable ground displacements
#import mask tiff
Mask_ground_df = pd.read_csv('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Shapefiles/KroBreKonBre_mask_stableground.csv')

#X corresponds to Easting[m] and Y to Northing[Y]
points_groundmask = list(zip(Mask_ground_df["X"].to_list(),Mask_ground_df["Y"].to_list()))
values_groundmask = Mask_ground_df["Mask"].to_list()

Disp_resolution = 48.5 # gives 350x350 output
xRange_mask = np.arange(Mask_ground_df["X"].min(),Mask_ground_df["X"].max(),Disp_resolution)
yRange_mask = np.arange(Mask_ground_df["Y"].min(),Mask_ground_df["Y"].max(),Disp_resolution)
#yRange_mask  = np.delete(yRange_mask, -1)

gridX_mask,gridY_mask = np.meshgrid(xRange_mask,yRange_mask )

#Raster of the ground mask with 1 (ground) and 0 (non ground) with the same resolution of FT output
grid_groundmask = griddata(points_groundmask, values_groundmask, (gridX_mask ,np.flip(gridY_mask) ), method='nearest')

grid_groundmask = grid_groundmask.astype(np.float64)
grid_groundmask[grid_groundmask==0] = np.nan

#%% Create masked displacement maps of stable ground
df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_groundmasked = []
df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_groundmasked = []
#also peakCorr for filtering using 0.5 threshold
df_peakCorr_allpairs_imported_groundmasked = []
df_peakCorr_singlepair_imported_groundmasked = []

#create dictionary with each FT output ground masked 
for i in range (33):
    df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_groundmasked = df_net_disp_meters_allpairs_imported_peakCorr_med_filtered[i] * grid_groundmask
    df_peakCorr_singlepair_imported_groundmasked = df_peakCorr_list[i] * grid_groundmask
    
    df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_groundmasked.append(df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_groundmasked )
    df_peakCorr_allpairs_imported_groundmasked.append(df_peakCorr_singlepair_imported_groundmasked)
    del df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_groundmasked
    del df_peakCorr_singlepair_imported_groundmasked

#%% Create string list for each pair
all_img_tiffile = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all')

datestring_all_img = []
datestring_single_img =[]

for i in range(34):
    datestring_single_img = all_img_tiffile[i][ 14: 22]
    datestring_all_img.append(datestring_single_img)

disp_ground_allpairs = []

# Load the dataframe
for i in range(33):
    #convert to numpy
    df_np = df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_groundmasked[i].to_numpy()    
    #convert in 1d
    df_np_1d = df_np.flatten(order = "C")
    #get just the non nan values
    filtered_data = df_np_1d[~np.isnan(df_np_1d)]
    #append to list
    disp_ground_allpairs.append(filtered_data)
    del df_np
    del df_np_1d
    del filtered_data

#disp_ground_allpairs.remove(16)
Disp_ground_avg_allpairs = []
Disp_ground_avg_singlepair = []

for i in range(33):
    Disp_ground_avg_singlepair = np.mean(disp_ground_allpairs[i])
    Disp_ground_avg_allpairs.append(Disp_ground_avg_singlepair)
    del Disp_ground_avg_singlepair

#%% Plot avg displacement on stable ground
plt.subplot(2,1,1)
plt.ylim(top=25, bottom = 0) 
plt.rcParams["figure.figsize"] = [12, 10]
plt.boxplot(disp_ground_allpairs, showfliers = False)
plt.ylabel("Average Displacement [m]", fontsize = 14) 
plt.xlabel("Date",  fontsize = 16) 
plt.yticks(fontsize = 14)
plt.ylim(top=2000, bottom = 0) 
defined_ticks = np.linspace(0, 33, num=34)
plt.xticks(ticks = defined_ticks ,
           labels = ['', '20170415', '20170505', '20170525', '20170729', '20180405', '20180410', '20180425', '20180808', '20180813', '20180917', '20190321', '20190331', '20190430', '20190505', '20190510', '20190525', '20190614', '20190704', '20190709', '20190803', '20190917', '20200330', '20200424', '20200623', '20200802', '20200807', '20210414', '20210429', '20210509', '20210519', '20210628', '20210802', '20210827'], 
           rotation = 50,
           fontsize = 14)

plt.subplot(2,1,2)
plt.ylim(top=25, bottom = 0) 
plt.rcParams["figure.figsize"] = [12, 10]
plt.boxplot(disp_ground_allpairs, showfliers = False)
plt.ylabel("Average Displacement [m]", fontsize = 14) 
plt.xlabel("Date",  fontsize = 16) 
plt.yticks(fontsize = 14)
plt.ylim(top=20, bottom = 0) 
defined_ticks = np.linspace(0, 33, num=34)
plt.xticks(ticks = defined_ticks ,
           labels = ['', '20170415', '20170505', '20170525', '20170729', '20180405', '20180410', '20180425', '20180808', '20180813', '20180917', '20190321', '20190331', '20190430', '20190505', '20190510', '20190525', '20190614', '20190704', '20190709', '20190803', '20190917', '20200330', '20200424', '20200623', '20200802', '20200807', '20210414', '20210429', '20210509', '20210519', '20210628', '20210802', '20210827'], 
           rotation = 50,
           fontsize = 14)

plt.tight_layout(pad=0.5)
plt.show()
#%%
plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Stable_ground/Stable_ground_displacement_both_versions.png', dpi = 500,  bbox_inches='tight')

#%% Create string list for each pair
all_img_tiffile = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all')

datestring_all_img = []
datestring_single_img =[]

for i in range(34):
    datestring_single_img = all_img_tiffile[i][ 14: 22]
    datestring_all_img.append(datestring_single_img)

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

df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked = df_net_disp_imported_list


#%% Get time difference of each pair to get velocities
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
Velocity_singlepair_imported_peakCorr_med_filtered_masked = []
Velocity_allpairs_imported_peakCorr_med_filtered_masked = []

for i in range(33):
    Velocity_singlepair_imported_peakCorr_med_filtered_masked = df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked[i] / timestep_days_allpairs[i]
    Velocity_allpairs_imported_peakCorr_med_filtered_masked.append(Velocity_singlepair_imported_peakCorr_med_filtered_masked)
    del Velocity_singlepair_imported_peakCorr_med_filtered_masked

xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_6md_x2/Velocity_allpairs_imported_peakCorr_med_filtered_masked.xlsx'
save_xls(Velocity_allpairs_imported_peakCorr_med_filtered_masked, xls_path)
del xls_path

#%%Save velocity rasters
Disp_resolution = 50
#define CRS
from rasterio.crs import CRS
rasterCrs = CRS.from_epsg(32633)

#definition of the raster transform array
from rasterio.transform import Affine
transform = Affine.translation(gridX_mask[0][0]-Disp_resolution, gridY_mask[0][len(gridY_mask)-1]-Disp_resolution)*Affine.scale(Disp_resolution,Disp_resolution)

points_glaciervelocity = []
values_glaciervelocity = []
grid_glaciervelocity = []

for i in range(33):
    points_glaciervelocity = list(zip(gridX_mask.flatten(),gridY_mask.flatten()))
    values_glaciervelocity = Velocity_allpairs_imported_peakCorr_med_filtered_masked[i].values.flatten()
    grid_glaciervelocity = griddata(points_glaciervelocity, values_glaciervelocity, (gridX_mask ,np.flip(gridY_mask) ), method='nearest')

    interpRaster = rasterio.open('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_tiff_img_velocities/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.tif',
                                'w',
                                driver='GTiff',
                                height=grid_glaciervelocity.shape[0],
                                width=grid_glaciervelocity.shape[1],
                                count=1,
                                dtype=grid_glaciervelocity.dtype,
                                crs=rasterCrs,
                                transform=transform,
                                )
    interpRaster.write(grid_glaciervelocity,1)
    interpRaster.close()
    
    del points_glaciervelocity
    del values_glaciervelocity
    del grid_glaciervelocity
#%% Plot displacement not filtered
#import images lat, lon
xls_path_lon = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw/Lon_interp_on_FT_results.csv'
xls_path_lat = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw/Lat_interp_on_FT_results.csv'

Lon_interp_on_FT_results = pd.read_csv(xls_path_lon)
Lat_interp_on_FT_results = pd.read_csv(xls_path_lat)

del Lon_interp_on_FT_results ["Unnamed: 0"]
del Lat_interp_on_FT_results ["Unnamed: 0"]

Lon_interp_on_FT_results = Lon_interp_on_FT_results.to_numpy()
Lat_interp_on_FT_results = Lat_interp_on_FT_results.to_numpy()

for i in range(33):
#plot
    plt.title("Sentinel-2 L1C Band8 \n" + "[" + datestring_all_img[i] + "]" + "-" + "["+ datestring_all_img[i+1] + "]", fontsize = 10)
    plt.xlabel("Easting [m]", fontsize = 10)
    plt.xticks(fontsize = 8)
    plt.ylabel("Northing [m]", fontsize = 10)
    plt.yticks(fontsize = 8)
    base_img_dir = 'C:/Users/39333/OneDrive - Universitetet i Oslo\Oslo_University/MSc_Thesis/Processed_data/Sentinel2/S2_b08_cropbox_KbKvKb_geotiff_all/S2B08_cropped_20170415.tif'
    base_img = rs.open(base_img_dir)
    ax = plt.pcolormesh(gridX_mask,np.flip(gridY_mask), df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked[i], alpha= 1, cmap='rainbow')
    cbar = plt.colorbar(ax, extend = 'both')
    rs.plot.show(base_img, cmap = 'gray')
    #check the following line, for plotting i take the 90th percentile and add 50m of disp
    plt.clim(0, 100) 
    cbar.set_label('Surface velocity [md$^{-1}$]', rotation=270, labelpad=15, fontsize = 10)
    cbar.minorticks_on()
    plt.rc('font', **{'size':'8'})

#Save fig
    plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Feature_Track_S2_snr_medfilt/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.png', dpi=300, bbox_inches='tight')

# Close open figures
    plt.close('all') 
#%% Plot displacement filtered peakCorr > 0.5 and 3x3 median filter
for i in range(33): 
    plt.title("Sentinel-2 L1C Band8 \n" + "[" + datestring_all_img[i] + "]" + "-" + "["+ datestring_all_img[i+1] + "]", fontsize = 10)
    plt.xlabel("Easting [m]", fontsize = 10)
    plt.xticks(fontsize = 8)
    plt.ylabel("Northing [m]", fontsize = 10)
    plt.yticks(fontsize = 8)
    plt.rc('font', **{'size':'8'})
#import base image
    base_img_dir = 'C:/Users/39333/OneDrive - Universitetet i Oslo\Oslo_University/MSc_Thesis/Processed_data/Sentinel2/S2_b08_cropbox_KbKvKb_geotiff_all/S2B08_cropped_20170415.tif'
    base_img = rs.open(base_img_dir)
#plot disp values filtered 
    levels = np.linspace(0, 250, 26)
    ax = plt.contourf(gridX_mask,np.flip(gridY_mask), df_net_disp_meters_allpairs_imported_peakCorr_med_filtered_masked[i], alpha= 1, cmap='rainbow', levels=levels, vmin = 0, vmax = 250)
#define colorbar
    cbar = plt.colorbar(ax, extend = 'both', label='Absolute surface displacement [m]')
    rs.plot.show(base_img, cmap = 'gray')
    cbar.set_label('Absolute surface displacement [m]', rotation=270, labelpad=15, fontsize = 10)


#Save fig
    plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Feature_Track_S2_snr_medfilt/Sameplots_with_contours/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.png', dpi=300, bbox_inches='tight')
# Close open figures
    plt.close('all') 
#%% Plot surface velocities
for i in range(33): 
    plt.title("Sentinel-2 L1C Band8 \n" + "[" + datestring_all_img[i] + "]" + "-" + "["+ datestring_all_img[i+1] + "]", fontsize = 10)
    plt.xlabel("Easting [m]", fontsize = 10)
    plt.xticks(fontsize = 8)
    plt.ylabel("Northing [m]", fontsize = 10)
    plt.yticks(fontsize = 8)
    plt.rc('font', **{'size':'8'})
#import base image
    base_img_dir = 'C:/Users/39333/OneDrive - Universitetet i Oslo\Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/S2B08_cropped_20170415.tif'
    base_img = rs.open(base_img_dir)
#plot disp values filtered 
    levels = np.linspace(0, 6, 13)
    ax = plt.contourf(gridX_mask,np.flip(gridY_mask), Velocity_allpairs_imported_peakCorr_med_filtered_masked[i], alpha= 1, cmap='rainbow', levels=levels, vmin = 0, vmax = 6)
#define colorbar
    cbar = plt.colorbar(ax, extend = 'both')
    rs.plot.show(base_img, cmap = 'gray')
    cbar.set_label('Surface velocity [md$^{-1}$]', rotation=270, labelpad=15, fontsize = 10)


#Save fig
    plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Feature_Track_S2_peakCorr_medfilt_velocities/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.png', dpi=300, bbox_inches='tight')
# Close open figures
    plt.close('all') 
    
    #%% Plot final velocities
    #import images lat, lon
    xls_path_lon = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw/Lon_interp_on_FT_results.csv'
    xls_path_lat = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw/Lat_interp_on_FT_results.csv'

    Lon_interp_on_FT_results = pd.read_csv(xls_path_lon)
    Lat_interp_on_FT_results = pd.read_csv(xls_path_lat)

    del Lon_interp_on_FT_results ["Unnamed: 0"]
    del Lat_interp_on_FT_results ["Unnamed: 0"]

    Lon_interp_on_FT_results = Lon_interp_on_FT_results.to_numpy()
    Lat_interp_on_FT_results = Lat_interp_on_FT_results.to_numpy()
#%% Plot final velocities with pcolormesh
    for i in range(33):
    #plot
        figure(figsize=(4,3))
        plt.title("[" + datestring_all_img[i] + "]" + "-" + "["+ datestring_all_img[i+1] + "]", fontsize = 8)
        plt.xlabel("Easting [m]", fontsize = 8)
        plt.xticks(fontsize = 6)
        plt.ylabel("Northing [m]", fontsize = 8)
        plt.yticks(fontsize = 6)
        base_img_dir = 'C:/Users/39333/OneDrive - Universitetet i Oslo\Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/S2B08_cropped_20170415.tif'
        base_img = rs.open(base_img_dir)
        ax = plt.pcolormesh(gridX_mask,np.flip(gridY_mask), Velocity_allpairs_imported_peakCorr_med_filtered_masked[i], alpha= 1, cmap='rainbow')
        cbar = plt.colorbar(ax, extend = 'both')
        rs.plot.show(base_img, cmap = 'gray')
        #check the following line, for plotting i take the 90th percentile
        plt.clim(0, 6) 
        cbar.set_label('Surface velocity [md$^{-1}$]', rotation=270, labelpad=15, fontsize = 8)
        cbar.minorticks_on()
        plt.rc('font', **{'size':'6'})

    #Save fig
        plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Feature_Track_S2_peakCorr_medfilt_velocities_v2/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.png', dpi=400, bbox_inches='tight')

    # Close open figures
        plt.close('all') 
    

    
