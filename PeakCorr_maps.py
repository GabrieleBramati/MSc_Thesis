# -*- coding: utf-8 -*-
"""

Plotting code for Sentinel 2 feature tracking - Peak Correlation maps
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
df_peakCorr_imported = pd.read_excel(xls_path_peakCorr, sheet_name=None)
for i in range(33):
    del df_peakCorr_imported[list(df_peakCorr_imported.keys())[i]]["Unnamed: 0"]
    
del xls_path_peakCorr

   
#%% #Convert imported dictionaries in list of dataframes
#get list of keys
keys = list(df_peakCorr_imported.keys())
    
df_peakCorr_list = []
for i in range(33):
    df = pd.DataFrame.from_dict(df_peakCorr_imported[keys[i]]) 
    df_peakCorr_list.append(df)
    del df

del df_peakCorr_imported

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


df_peakCorr_imported_masked_single = []
df_peakCorr_imported_masked_all = []

#create dictionary with each FT output masked 
for i in range (33):
    df_peakCorr_imported_masked_single = df_peakCorr_list[i] * grid_glaciermask
    df_peakCorr_imported_masked_all.append(df_peakCorr_imported_masked_single)
    del df_peakCorr_imported_masked_single
#%% 
single_max = []
coeff_peakCorr_single = []
normalized_peakCorr_single = []
normalized_peakCorr_all = []

for i in range (33):
    single_max = df_peakCorr_imported_masked_all[i].max().max()
    coeff_peakCorr_single = (df_peakCorr_imported_masked_all[i] / single_max) * 100
    normalized_peakCorr_single = df_peakCorr_imported_masked_all[i].div(coeff_peakCorr_single)
    normalized_peakCorr_all.append(normalized_peakCorr_single)
    del single_max
    del coeff_peakCorr_single
    del normalized_peakCorr_single

#%%

from sklearn import preprocessing
single_values = []
min_max_scaler = []
single_values_scaled = []
single_df = []
normalized_peakCorr_all = []

for i in range (33):
    single_values = df_peakCorr_imported_masked_all[i].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    single_values_scaled = min_max_scaler.fit_transform(single_values)
    single_df = pd.DataFrame(single_values_scaled)
    normalized_peakCorr_all.append(single_df)
    del single_values
    del min_max_scaler
    del single_values_scaled
    del single_df
#%%
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
    
#%% Plot peak Correlation maps
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
    levels = np.linspace(0, 1, 11)
    ax = plt.contourf(gridX_mask,np.flip(gridY_mask), normalized_peakCorr_all[i], alpha= 1, cmap='rainbow', levels=levels, vmin = 0, vmax = 1)
#define colorbar
    cbar = plt.colorbar(ax, extend = 'both')
    rs.plot.show(base_img, cmap = 'gray')
    cbar.set_label('Peak Correlation Coeff. [-]', rotation=270, labelpad=15, fontsize = 10)


#Save fig
    plt.savefig('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Figures/Feature_Track_S2_peakCorr_maps/' + datestring_all_img[i] + '_' + datestring_all_img[i+1] + '.png', dpi=300, bbox_inches='tight')
# Close open figures
    plt.close('all') 