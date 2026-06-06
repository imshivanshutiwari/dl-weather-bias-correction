#The following code file will help to train the CNNBC Model as well as produce the bias corrected results for both training duration and the testing duration.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gc
import matplotlib.pyplot as plt

import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]= "7"

tx= np.load('/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/npy_files/X_train_09_07.npy')
tsx=np.load('/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/npy_files/X_test_09_07.npy')
ty=np.load('/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/y_train_09_07.npy')
tsy=np.load('/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/y_test_09_07.npy')

def create_model():
  input_layer = layers.Input(shape=(10,32,32,1))
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
    filepath="checkmodel.keras",
    monitor='val_loss',
    verbose=1,
    mode='min',
    save_best_only= True,
    )
  return model,checkpoint,earlystop


path="/home/hpcs_rnd/Yatendra_IITKgp/trash/"
model_CNNBC,checkpoint,earlystop = create_model()
history=model_CNNBC.fit(tx,ty,batch_size=16,epochs=500, verbose=2, validation_split=0.20,callbacks=[checkpoint, earlystop])
model_CNNBC.save(path+"my_model.h5", include_optimizer=True)

tr_result=model_CNNBC.predict(tx)
ts_result=model_CNNBC.predict(tsx)

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

    plt.show()

plot_history(history)

tr_result_flat = tr_result.flatten()
label_flat = ty.flatten()
ts_result_flat = ts_result.flatten()
tsx_flat = tsy.flatten()

# Calculate correlation coefficients
# train_correlation = np.corrcoef(tr_result_flat, label_flat)[0, 1]
test_correlation = np.corrcoef(ts_result_flat, tsx_flat)[0, 1]
print('correlation is:',test_correlation)
