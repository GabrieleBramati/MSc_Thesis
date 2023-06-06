#Code to import S2 images and perform feature tracking for each pair and save

## SET Working directory
import os
os.chdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Codes')

#%% Import packages needed
from templatematch import templatematch
from geoimread import geoimread
import pandas as pd
import numpy as np
#%% Create list of images
base_dir_list = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/'
S2L1C_dir_list = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all/')

#%% Create directory for each image
def absolute_file_paths(base_dir_list):
    path = os.path.abspath(base_dir_list)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]

images_dir_list = absolute_file_paths(base_dir_list)
#%% Import all images
images_num_list = {}
for i in range(1,len(images_dir_list) + 1,1):
    img = str("img"+str(i))
    images_num_list[i] = str("img"+str(i))

# list of image numbers (ex: img1, img2, etc)
images_num_list = list(images_num_list.values())

# import img files with geoimread
imported_img = []
for i in range(34):
    img = geoimread(images_dir_list[i], roi_x= None, roi_y= None, roi_crs={'init': 'EPSG:32633'}, buffer= None)
    imported_img.append(img)

#%% Create string list for each pair
all_img_tiffile = os.listdir('C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Sentinel2_cropped_b08/S2_b08_cropbox_KbKvKb_geotiff_all')

datestring_all_img = []
datestring_single_img =[]

for i in range(34):
    datestring_single_img = all_img_tiffile[i][ 14: 22]
    datestring_all_img.append(datestring_single_img)

#%% Get time time of each image and timestep for pairs
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

#Generate max expected displacement for each pair multiplying the
#timestep of each pair with the max summer velocity (6 m/d) obtained by a random
#run with big search window 
max_expected_disp = [i * 6 for i in timestep_days_allpairs]
#convert the max expected displacement into pixels by dividing by 10
max_expected_disp_pix = [i / 10 for i in max_expected_disp]

#multiply the max expected disp x2
max_expected_disp_pix_x2= []
max_expected_disp_pix_x2_single = []

for i in range (32):
    #do the sum for the validation pair between the two sequent images timestep
    max_expected_disp_pix_x2_single = max_expected_disp_pix[i]*2 + max_expected_disp_pix[i+1]*2
    max_expected_disp_pix_x2.append(max_expected_disp_pix_x2_single)
    del max_expected_disp_pix_x2_single
 
max_expected_disp_pix_x2 = np.array(max_expected_disp_pix_x2)
max_expected_disp_pix_x2 = max_expected_disp_pix_x2.astype(int)

#%% Perform feature tracking on all pairs

#One step pairs
featuretrack_output_raw = []
df_pu = [] #pixels longitude
df_pv = [] #pixels latitude
df_du = [] #pixels displacement longitude
df_dv = [] #pixels displacement latitude
df_peakCorr = [] #peak correlation
df_meanAbs_Corr = [] #mean absolute correlation
df_snr = [] #signal to noise ratio

pu = np.linspace(0, 1750, num=350, endpoint=False)
pv = np.linspace(0, 1750, num=350, endpoint=False)
pu, pv = np.meshgrid(pu, pv)

for i in range(32): #validation 2 steps images
    ft_pair = templatematch(imported_img[i], imported_img[i+2], pu=pu, pv=pv, TemplateWidth=11, SearchWidth= 11+ max_expected_disp_pix_x2[i])
    featuretrack_output_raw.append(ft_pair)
    df_pu.append(pd.DataFrame(ft_pair.pu))
    df_pv.append(pd.DataFrame(ft_pair.pv))
    df_du.append(pd.DataFrame(ft_pair.du))
    df_dv.append(pd.DataFrame(ft_pair.dv))
    df_peakCorr.append(pd.DataFrame(ft_pair.peakCorr))
    df_meanAbs_Corr.append(pd.DataFrame(ft_pair.meanAbsCorr))
    df_snr.append(pd.DataFrame(ft_pair.snr))
    del ft_pair

#%%create function to save files
from pandas import ExcelWriter

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()
        
#%% save each list of dataframe output

#save df_pu
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_pu_valpairs.xlsx'
save_xls(df_pu, xls_path)
del xls_path

#save df_pv
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_pv_valpairs.xlsx'
save_xls(df_pv, xls_path)
del xls_path

#save df_du
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_du_valpairs.xlsx'
save_xls(df_du, xls_path)
del xls_path

#save df_dv
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_dv_valpairs.xlsx'
save_xls(df_dv, xls_path)
del xls_path

#save df_meanAbsCorr
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_meanAbs_Corr_valpairs.xlsx'
save_xls(df_meanAbs_Corr, xls_path)
del xls_path

#save df_peakCorr
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_peakCorr_valpairs.xlsx'
save_xls(df_peakCorr, xls_path)
del xls_path

#save df_snr
xls_path = 'C:/Users/39333/OneDrive - Universitetet i Oslo/Oslo_University/MSc_Thesis/Processed_data/Output_featuretracking_raw_adaptivesearch_valpairs_6md_x2/df_snr_valpairs.xlsx'
save_xls(df_snr, xls_path)
del xls_path

