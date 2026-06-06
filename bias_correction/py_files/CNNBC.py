#The following code file will help to train the CNNBC Model as well as produce the bias corrected results for both training duration and the testing duration.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gc


train_year=np.load("./Data/YearIndex500Train.npy")
test_year=np.load("./Data/YearIndex500Test.npy")

data=np.load("./Data/inputdata100.npy") # Load the input samples for the model(biased data samples) 
label=np.load("./Data/label100.npy") # Load the labels(observation samples)
#biasdata=np.load("./Data/biasdata100.npy") 


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
  model.compile(optimizer='adam', loss='mean_squared_error')
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
    filepath="checkmodel",
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
    tr_year=train_year[i][j]
    start_index_tr=(tr_year)*43
    end_index_tr=(tr_year+1)*43
    for l in range (start_index_tr,end_index_tr):
      trainx.append(data[l])
      trainy.append(label[l])
   #   trainb.append(biasdata[l])
  for k in range(0,6):
    ts_year=test_year[i][k]
    start_index_ts=(ts_year)*43
    end_index_ts=(ts_year+1)*43
    for m in range (start_index_ts,end_index_ts):
      testx.append(data[m])
      testy.append(label[m])
    #  testb.append(biasdata[m])
  tx=tf.convert_to_tensor(trainx)
  ty=tf.convert_to_tensor(trainy)
  tsx=tf.convert_to_tensor(testx)
  path="/scratch/19ai91r07/"
  model_CNNBC,checkpoint,earlystop = create_model()
  model_CNNBC.fit(tx,ty,batch_size=16,epochs=500, verbose=2, validation_split=0.20,callbacks=[checkpoint, earlystop])
  model_CNNBC.save(path+"/Model/CNNBC"+str(i))

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

  np.save(path+"/Result/tr_result"+str(i)+".npy",rtrain)
  np.save(path+"/Result/ts_result"+str(i)+".npy",rtest)
  np.save(path+"/Result/tr_label"+str(i)+".npy",ltrain)
  np.save(path+"/Result/ts_label"+str(i)+".npy",ltest)
  #np.save(path+"/Result/tr_bias"+str(i)+".npy",btrain)
  #np.save(path+"/Result/ts_bias"+str(i)+".npy",btest)

  del model_CNNBC,checkpoint,earlystop
  del tr_result,ts_result
  del tx,ty,tsx
  del trainx,trainy,trainb,testx,testy,testb
  del tr_year,ts_year
  del start_index_tr,end_index_tr,start_index_ts,end_index_ts
  del rtrain,rtest,ltrain,ltest,btrain,btest
  gc.collect()

