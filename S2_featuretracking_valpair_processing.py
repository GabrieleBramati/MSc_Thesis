#Code to import feature tracking results for each pair for VALIDATION

## SET Working directory
import os
os.chdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Codes')

#%% Import packages needed
from matplotlib import pyplot as plt
from pandas import ExcelWriter
#import rasterio as rs 
#import rasterio.plot
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
#from pyproj import Proj
from scipy.interpolate import griddata
#from scipy.interpolate import RectBivariateSpline
from scipy import signal
#from osgeo import gdal
#from osgeo import gdalconst

#%%Define excel save 
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()
        
#%% Import all tables output from feature tracking

#Define path to files from block above
xls_path_pu_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_pu_valpairs.xlsx'
xls_path_pv_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_pv_valpairs.xlsx'
xls_path_du_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_du_valpairs.xlsx'
xls_path_dv_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_dv_valpairs.xlsx'
xls_path_meanAbsCorr_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_meanAbs_Corr_valpairs.xlsx'
xls_path_peakCorr_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_peakCorr_valpairs.xlsx'
xls_path_snr_val = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_snr_valpairs.xlsx'
    
df_pu_val = pd.read_excel(xls_path_pu_val, sheet_name=None)
sheets = df_pu_val.keys()
for sheet in sheets:
    df_pu_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_pu_val[sheet]["Unnamed: 0"]
    
df_pv_val = pd.read_excel(xls_path_pv_val, sheet_name=None)
sheets = df_pv_val.keys()
for sheet in sheets:
    df_pv_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_pv_val[sheet]["Unnamed: 0"]

df_du_val = pd.read_excel(xls_path_du_val, sheet_name=None)
sheets = df_du_val.keys()
for sheet in sheets:
    df_du_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_du_val[sheet]["Unnamed: 0"]

df_dv_val = pd.read_excel(xls_path_dv_val, sheet_name=None)
sheets = df_dv_val.keys()
for sheet in sheets:
    df_dv_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_dv_val[sheet]["Unnamed: 0"]

df_meanAbsCorr_val = pd.read_excel(xls_path_meanAbsCorr_val, sheet_name=None)
sheets = df_meanAbsCorr_val.keys()
for sheet in sheets:
    df_meanAbsCorr_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_meanAbsCorr_val[sheet]["Unnamed: 0"]
    
df_peakCorr_val = pd.read_excel(xls_path_peakCorr_val, sheet_name=None)
sheets = df_peakCorr_val.keys()
for sheet in sheets:
    df_peakCorr_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_peakCorr_val[sheet]["Unnamed: 0"]

df_snr_val = pd.read_excel(xls_path_snr_val, sheet_name=None)
sheets = df_snr_val.keys()
for sheet in sheets:
    df_snr_val[sheet].to_excel(f"{sheet}.xlsx")
    del df_snr_val[sheet]["Unnamed: 0"]

#Convert imported dictionaries in list of dataframes
#get list of keys
keys = list(df_pu_val.keys())

df_pu_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_pu_val[keys[i]]) 
    df_pu_val_imported.append(df)
    del df

del df_pu_val

df_pv_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_pv_val[keys[i]]) 
    df_pv_val_imported.append(df)
    del df

del df_pv_val

df_du_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_du_val[keys[i]]) 
    df_du_val_imported.append(df)
    del df

del df_du_val

df_dv_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_dv_val[keys[i]]) 
    df_dv_val_imported.append(df)
    del df
    
del df_dv_val

df_meanAbsCorr_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_meanAbsCorr_val[keys[i]]) 
    df_meanAbsCorr_val_imported.append(df)
    del df
    
del df_meanAbsCorr_val
    
df_peakCorr_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_peakCorr_val[keys[i]]) 
    df_peakCorr_val_imported.append(df)
    del df

del df_peakCorr_val
    
df_snr_val_imported = []
for i in range(32):
    df = pd.DataFrame.from_dict(df_snr_val[keys[i]]) 
    df_snr_val_imported.append(df)
    del df

del df_snr_val

#%% Compute displacement for each pair
net_disp_pixels = []
net_disp_meters = []
net_disp_meters_val = []

for i in range(32): #images are 34, therefore validation pairs are 32
    net_disp_pixels = np.sqrt(df_du_val_imported[i] ** 2 + df_dv_val_imported[i] ** 2)
    net_disp_meters = net_disp_pixels.multiply(10)
    net_disp_meters_val.append(net_disp_meters)
    del net_disp_pixels
    del net_disp_meters

# =============================================================================
# xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_validation_pairs/net_disp_meters_val.xlsx'
# save_xls(net_disp_meters_val, xls_path)
# del xls_path
# =============================================================================

#%% Filter all displacement raster for Max correlation and apply median filter with a 3x3 kernel size
df_net_disp_meters_valpairs_imported_peakCorr_med_filtered = []
single_pair_peakCorr_filtered = []
single_pair_peakCorr_filtered_medfilt = []

for i in range(32):
#Define peakCorr threshold
    peakCorr_index = df_peakCorr_val_imported[i].loc[:,:] > 50
#Convert "False" to nan
    peakCorr_index = peakCorr_index * 1
    peakCorr_index[peakCorr_index==0] = np.nan
#Multiply the displacement raster to where the peakCorr threshold is satisfied
    single_pair_peakCorr_filtered = net_disp_meters_val[i] * peakCorr_index
#Filtering with 3x3 medial filter    
    single_pair_peakCorr_filtered_medfilt = signal.medfilt2d(single_pair_peakCorr_filtered, kernel_size=3)
    single_pair_peakCorr_filtered_medfilt = pd.DataFrame(single_pair_peakCorr_filtered_medfilt)
#Final dataframe with snr and 3x3 median filter    
    df_net_disp_meters_valpairs_imported_peakCorr_med_filtered.append(single_pair_peakCorr_filtered_medfilt)
    del peakCorr_index
    del single_pair_peakCorr_filtered
    del single_pair_peakCorr_filtered_medfilt

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

df_net_disp_meters_valpairs_imported_peakCorr_med_filtered_masked = []
df_net_disp_meters_singlevalpair_imported_peakCorr_med_filtered_masked = []

#create dictionary with each FT output masked 
for i in range (32):
    df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked = df_net_disp_meters_valpairs_imported_peakCorr_med_filtered[i] * grid_glaciermask
    df_net_disp_meters_valpairs_imported_peakCorr_med_filtered_masked.append(df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked)
    del df_net_disp_meters_singlepair_imported_peakCorr_med_filtered_masked


xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Triangulation_input/df_net_disp_meters_valpairs_imported_peakCorr_med_filtered_masked.xlsx'
save_xls(df_net_disp_meters_valpairs_imported_peakCorr_med_filtered_masked, xls_path)
del xls_path

#END OF PROCESSING PART
    







#%%
#create Easting, Northing, Displacement dataframes list for each validation pair
net_disp_meters_1d_allpairs = []
net_disp_meters_1d = []
net_disp_meters_1d_np = []

#for loop to create 1d displacement
for i in range (33):
    net_disp_meters_1d_np = df_net_displacements_allpairs_snr_3medifilt_listofdf_300x300[i]
    net_disp_meters_1d = net_disp_meters_1d_np.flatten(order = "C")
    net_disp_meters_1d_allpairs.append(net_disp_meters_1d) 
    del net_disp_meters_1d_np
    del net_disp_meters_1d

#get Lat Lon of validation pairs
lon_interp_val = pd.DataFrame(interp1d(np.arange(0, imported_img[1].x.shape[0]), imported_img[1].x, fill_value="extrapolate")(df_pu_val[1]))
lat_interp_val = pd.DataFrame(interp1d(np.arange(0, imported_img[1].y.shape[0]), imported_img[1].y, fill_value="extrapolate")(df_pv_val[1]))

lon_interp_val =  lon_interp_val.to_numpy()
lat_interp_val =  lat_interp_val.to_numpy()

lon_interp_val_1d = lon_interp_val.flatten(order = "C")
lat_interp_val_1d = lat_interp_val.flatten(order = "C")

Disp_df_allpairs = []
Disp_df_single = []

#for loop to add easting and northing to the displacements
for i in range(33):
    Disp_df_single = np.stack((lon_interp_val_1d,lat_interp_val_1d, net_disp_meters_1d_allpairs[i]), axis=1)
    Disp_df_single = pd.DataFrame(Disp_df_single)
    Disp_df_single.columns = ['Easting(m)', 'Northing(m)', 'Displacement(m)']
    Disp_df_allpairs .append(Disp_df_single)
    del Disp_df_single


#%%Create final Grid with lat,lon and Displacement values
#define raster resolution
points = []
values = []
gridX = []
gridY = []
gridDisp = []
gridDisp_allpairs = []

#create coord ranges over the desired raster extension. Same for all, not in the for loop
xRange = lon_interp_val[0,:]
yRange = lat_interp_val[:,0]
#yRange = np.flip(yRange)

points = []
values = []
gridX = []
gridY = []
gridDisp = []
gridDisp_allpairs_300x300 = []

for i in range(33):
    points = list(zip(lon_interp_val_1d,lat_interp_val_1d))
    values = Disp_df_allpairs[i]["Displacement(m)"].to_list()
    #create arrays of x,y over the raster extension
    gridX,gridY = np.meshgrid(xRange,yRange)
    #interpolate over the grid. Raster with Displacement values
    gridDisp = griddata(points, values, (gridX,gridY), method='nearest')
    gridDisp_allpairs_300x300.append(pd.DataFrame(gridDisp))
    del points
    del values
    del gridX
    del gridY
    del gridDisp

xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_validation_pairs/GridDisp_df_allpairs.xlsx'
save_xls(gridDisp_allpairs, xls_path)

#%%
Disp_df_allpairs[2].plot(x="Easting(m)", y="Northing(m)", kind="scatter", c="Displacement(m)",
        colormap="RdBu")

#%% Quick plot
s = 20
im = plt.imshow(gridDisp_allpairs_300x300[s], cmap="bwr")
plt.colorbar(im)
plt.show()
