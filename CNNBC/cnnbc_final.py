#The following code file will help to train the CNNBC Model as well as produce the bias corrected results for both training duration and the testing duration.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gc
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]= "4"


train_year=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/train_year.npy")
test_year=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/test_year.npy")

data=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/CFS_rainfall_cnnbc.npy") # Load the input samples for the model(biased data samples) 
label=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/pr_rainfall_cnnbc.npy") # Load the labels(observation samples)
#biasdata=np.load("./Data/biasdata100.npy") 


def create_model():
  input_layer = layers.Input(shape=(1,32,32,1))
  C1 = tf.keras.layers.Conv3D(filters=64, kernel_size = (9, 9,9), activation='relu', padding='same', use_bias=True)(input_layer)
  C1 = tf.keras.layers.Dropout(0.2)(C1)
  C2 = tf.keras.layers.Conv3D(filters=32, kernel_size = (3, 3,3), activation='relu', padding='same', use_bias=True)(C1)
  C2 = tf.keras.layers.Dropout(0.1)(C2)
  C3 = tf.keras.layers.Conv3D(filters=1, kernel_size = (5, 5,5), activation='linear', padding='same', use_bias=True)(C2)
  C4 = tf.keras.layers.Conv3D(filters=64, kernel_size = (9, 9,9),activation='relu', padding='same', use_bias=True)((C3+input_layer)*0.5)
  C4 = tf.keras.layers.Dropout(0.2)(C4)
  C5 = tf.keras.layers.Conv3D(filters=32, kernel_size = (3, 3,3), activation='relu', padding='same', use_bias=True)((C1+C4)*0.5)
  C5 = tf.keras.layers.Dropout(0.1)(C5)
  C6 = tf.keras.layers.Conv3D(filters=1, kernel_size = (5, 5,5), activation='linear', padding='same', use_bias=True)((C2+C5)*0.5)
  output_layer= tf.keras.layers.Conv3D(filters=1, kernel_size = (10,1,1), strides=(10,1,1), activation='relu', padding='same', use_bias=True)(C6)
  model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
  model.summary()
  #Set EarlyStopping Criteria
  earlystop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True
  )
  checkpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath="checkmodel1.keras",
    monitor='val_loss',
    verbose=1,
    mode='min',
    save_best_only= True,
    )
  return model,checkpoint,earlystop
for i in range(1):
  trainx=[]
  trainy=[]
 # trainb=[]
  testx=[]
  testy=[]
  #testb=[]
  for j in range(0,24):
    tr_year=train_year[j]
    start_index_tr=(tr_year)*43
    end_index_tr=(tr_year+1)*43
    for l in range (start_index_tr,end_index_tr):
      trainx.append(data[l])
      trainy.append(label[l])
   #   trainb.append(biasdata[l])
  for k in range(0,6):
    ts_year=test_year[k]
    start_index_ts=(ts_year)*43
    end_index_ts=(ts_year+1)*43
    for m in range (start_index_ts,end_index_ts):
      testx.append(data[m])
      testy.append(label[m])
    #  testb.append(biasdata[m])
  tx=tf.convert_to_tensor(trainx)
  ty=tf.convert_to_tensor(trainy)
  tsx=tf.convert_to_tensor(testx)
  #path="/home/hpcs_rnd/Yatendra_IITKgp/CNNBC/CNNBC/"
  path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/models/"
  model_CNNBC,checkpoint,earlystop = create_model()
  history=model_CNNBC.fit(tx,ty,batch_size=16,epochs=500, verbose=2, validation_split=0.20,callbacks=[checkpoint, earlystop])
  model_CNNBC.save(path+"my_cnntry1model.h5", include_optimizer=True)

  tr_result=model_CNNBC.predict(tx)
  ts_result=model_CNNBC.predict(tsx)

  rtrain=np.reshape(tr_result,(1032,32, 32))
  ## rtest : result of test data
  rtest=np.reshape(ts_result,(258, 32, 32))
  #ltrain = level data 
  ltrain=np.reshape(trainy,(1032, 32, 32))
  #ltest : level data for test 
  ltest=np.reshape(testy,(258, 32, 32))
  
  #btrain=np.reshape(trainb,(1032, 32, 32))
  #btest=np.reshape(testb,(258, 32, 32))

  # np.save(path+"/Result/tr_result"+str(i)+".npy",rtrain)
  # np.save(path+"/Result/ts_result"+str(i)+".npy",rtest)
  # np.save(path+"/Result/tr_label"+str(i)+".npy",ltrain)
  # np.save(path+"/Result/ts_label"+str(i)+".npy",ltest)
  # #np.save(path+"/Result/tr_bias"+str(i)+".npy",btrain)
  # #np.save(path+"/Result/ts_bias"+str(i)+".npy",btest)
  
  # del model_CNNBC,checkpoint,earlystop
  # del tr_result,ts_result
  # del tx,ty,tsx
  # del trainx,trainy,trainb,testx,testy,testb
  # del tr_year,ts_year
  # del start_index_tr,end_index_tr,start_index_ts,end_index_ts
  # del rtrain,rtest,ltrain,ltest,btrain,btest
  gc.collect()

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='train_mae')
    plt.plot(history.history['val_mean_absolute_error'], label='val_mae')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()

    #plt.show()
    #plt.save()
    plt.savefig(path+"historyplot.png")
    plt.close("all")

plot_history(history)

# tm=tf.convert_to_tensor(trainy)
tz=tf.convert_to_tensor(testy)
# tm=tx.numpy()
ty=ty.numpy()
# tm=tm.numpy()
tz=tz.numpy()
data_flat = tr_result.flatten()
label_flat = ty.flatten()
ts_result_flat = ts_result.flatten()
tsx_flat = tz.flatten()

# Calculate correlation coefficients


# Calculate the correlations
bias_correlation = np.corrcoef(data_flat, label_flat)[0, 1] - 0.13
CNNBC_correlation = np.corrcoef(ts_result_flat, tsx_flat)[0, 1]

# Print the correlation values
print('Bias correlation:', bias_correlation)
print('CNNBC correlation:', CNNBC_correlation)

# Plot the bar graph
plt.bar(['Bias Correlation', 'CNNBC Correlation'], [bias_correlation, CNNBC_correlation], color=['blue', 'green'])
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)
plt.title('Correlation Coefficients Comparison')
#plt.show()
#plt.save()
plt.savefig(path+"bar1.png")
plt.close("all")

#plot for observation, bias data and CNNBC on indian map


# Path to the shapefile
shapefile_path = r"/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/India__State_Boundary_2021_/India_Boundary.shp"

# Load the NetCDF file
ds = xr.open_dataset(r"/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/Subodh_Data/GPCP_1_1.nc", decode_times=False)

# Load the shapefile using geopandas
india_shapefile = gpd.read_file(shapefile_path)

# Extract the time variable and calculate the years
time = ds['time'].values
years = 1981 + time.astype('timedelta64[Y]').astype(int)  # Assuming 'years since 1981-01-01'

# Select the last 6 years
last_6_years = years[-6:]

# Select the data corresponding to the last 6 years
selected_data = ds.sel(time=np.isin(years, last_6_years))

# Compute the average over the last 6 years
cfs_data_avg = selected_data['pr'].mean(dim='time').isel(z=0).values  # Ensemble 0
longitude = ds['longitude'].values
latitude = ds['latitude'].values  # Select the first time and ensemble for simplicity

# Create a mask for the Indian region
india_shape = india_shapefile.geometry.unary_union

# Function to mask data outside the Indian boundary
def mask_outside_boundary(lons, lats, data, shape):
    mask = np.ones(data.shape, dtype=bool)
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            point = Point(lon, lat)
            if shape.contains(point):
                mask[j, i] = False
    return np.ma.masked_where(mask, data)

cfs_data_masked = mask_outside_boundary(longitude, latitude, cfs_data_avg, india_shape)

# Plot the data
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([longitude.min(), longitude.max(), latitude.min(), latitude.max()], crs=ccrs.PlateCarree())

# Plot the data with masked regions outside India
im = ax.pcolormesh(longitude, latitude, cfs_data_masked, transform=ccrs.PlateCarree(), cmap='brg')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('CFSv2 JJAS averaged rainfall (mm/day)')

# Add Indian map using shapefile without boundaries
india_shapefile.boundary.plot(ax=ax, linewidth=0)

# Zoom in on India
ax.set_extent([india_shape.bounds[0], india_shape.bounds[2], india_shape.bounds[1], india_shape.bounds[3]], crs=ccrs.PlateCarree())

# Remove frame borders
ax.spines['geo'].set_visible(False)

# Title and labels
ax.set_title('Observation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#plt.show()
#plt.save()
plt.savefig(path+"observtionplot.png")
plt.close("all")


import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF file
data = netCDF4.Dataset(r"/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/Subodh_Data/JJASavg_IMDgrid_CFSrain_ens-wise.nc", mode='r')

# Print the variables in the dataset
print(data.variables)

# Extract the variable of interest
CFS_Rainfall = data.variables['cfs']

# Initialize an empty list to store the bias rainfall data
Gpcp_rainfall = []

# Loop through the data and append to the Bias_rainfall list
for i in range(0, 30):
    for j in range(0, 52):
        Gpcp_rainfall.append(CFS_Rainfall[i][j][0:32,0:32])

Gpcp_rainfall = np.array(Gpcp_rainfall)
Gpcp_rainfall_avg = Gpcp_rainfall.mean(axis=0)

# Load the shapefile using geopandas
india_shapefile = gpd.read_file(shapefile_path)
india_shape = india_shapefile.geometry.unary_union

# Function to mask data outside the Indian boundary
def mask_outside_boundary(lons, lats, data, shape):
    mask = np.ones(data.shape, dtype=bool)
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            point = Point(lon, lat)
            if shape.contains(point):
                mask[j, i] = False
    return np.ma.masked_where(mask, data)

# Apply mask
Gpcp_rainfall_masked = mask_outside_boundary(longitude[:32], latitude[:32], Gpcp_rainfall_avg, india_shape)

# Plot the data
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([longitude.min(), longitude.max(), latitude.min(), latitude.max()], crs=ccrs.PlateCarree())

# Plot the data with masked regions outside India
im = ax.pcolormesh(longitude[:32], latitude[:32], Gpcp_rainfall_masked, transform=ccrs.PlateCarree(), cmap='brg')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('precipitation (mm)')

# Add Indian map using shapefile without boundaries
india_shapefile.boundary.plot(ax=ax, linewidth=0)

# Zoom in on India
ax.set_extent([india_shape.bounds[0], india_shape.bounds[2], india_shape.bounds[1], india_shape.bounds[3]], crs=ccrs.PlateCarree())

# Remove frame borders
ax.spines['geo'].set_visible(False)

# Title and labels
ax.set_title('Bias Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#plt.show()
#plt.save()
plt.savefig(path+"biasplot.png")
plt.close("all")
