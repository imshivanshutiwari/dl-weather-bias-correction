import numpy as np




data=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/CFS_rainfall_cnnbc.npy") # Load the input samples for the model(biased data samples) 
#label=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/pr_rainfall_cnnbc.npy") # Load the labels(observation samples)
label=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/gpcp_32x32.npy")

d2=[]
l2=[]
for i in range(30):
    for j in range(43):
        d2.append(data[i*52+j:i*52+j+10,0,:,:,:])
        l2.append(label[i:i+10,:,:,:])
    
d2=np.array(d2)
l2=np.array(l2)

np.save("data_10x32x32.npy",d2)
np.save("label_10x32x32.npy",l2)