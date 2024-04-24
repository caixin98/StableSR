#mv data from data/flatnet/real_captures to data/flatnet_val/real_captures if the filename is in data/flatnet_val/gts 
import os
import shutil
import numpy as np

def mv_data_from_flatnet():
    #path to the data
    path = '/root/caixin/StableSR/data/flatnet2single/inputs'
    path_val = 'data/flatnet_val/inputs'
    path_gts = 'data/flatnet_val/gts'
    os.makedirs(path_val, exist_ok=True)
    #list of the files in the gts directory
    gts = os.listdir(path_gts)
    #list of the files in the real_captures directory
    files = os.listdir(path)
    #loop through the files in the real_captures directory
    for file in files:
        #if the file is in the gts directory
        if file in gts:
            #move the file to the real_captures directory
            shutil.copy(os.path.join(path,file), os.path.join(path_val,file))
            print('copyed', file)
    print('done')
if __name__ == '__main__':
    mv_data_from_flatnet()  #run the function