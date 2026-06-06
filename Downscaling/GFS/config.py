CONFIG = {
            "GPU_NUM" : "1",
            "BATCH_SIZE" : 512,  
            "NUM_EPOCHS" : 15000,
            "EVAL_EVERY" : 50,
            "LEARNING_RATE" : 1e-06,
            "MOMENTUM" : 0.9,
            "PATIENCE" : 10,
            "FACTOR" : 0.1,
            "DISC_LEAD":0,

            
            "DATA_PATH" : "/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23_05deg.nc",
            "LABEL_PATH" : "/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc",
            "ORO_PATH" : "IMD_DATA/ORO_SCALED.nc",

            "CHECKPOINT_PATH" : "checkpoints/"
            
          }