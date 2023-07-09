
import os

import mat73
from random import randint
import numpy as np
import time
import pickle
from copy import deepcopy

# pylint: disable-all
class Data_prep:
    
    def __init__(self, data_path):
        
        
        self.data_path = data_path
        self.save_path = "data/"
        self.load_data()
        
        
    def load_data(self):
        
        print("Loading Data")
        
        MODIS_AOD_file= mat73.loadmat(os.path.join(self.data_path,\
            "Regridded_MODIS_AOD_4km_Mid_2017_2022_CONUS.mat"))
        CMAQ_PM25 = mat73.loadmat(os.path.join(self.data_path,\
            "Regridded_CMAQ_PM2.5_4km_Feb_June_2019_2020_CONUS_NaN_Filled.mat"))
        PUS4k = mat73.loadmat(os.path.join(self.data_path,"Regridded_PUS_4km_2016_CONUS.mat"))
        # humidity = mat73.loadmat()
        
        
        print('norm with max')
        cmaq_max = 210
        self.cmaq = np.einsum('ijk->kij', CMAQ_PM25['CMAQ_PM_All_Days_Filled'])
        print(self.cmaq.max())
        self.cmaq = self.cmaq/cmaq_max
        self.modis = np.einsum('ijk->kij', MODIS_AOD_file['MODIS_AOD'])
        modis_max = np.nanmax(self.modis)
        self.modis = self.modis/modis_max
        self.pus = PUS4k['PUS_CONUS']
        pus_max = np.nanmax(self.pus)
        self.pus = self.pus/pus_max
        
        
        max_values = [cmaq_max, modis_max, pus_max]
        file_ = os.path.join(self.save_path, 'maxes.pkl')
        with open(file_, 'wb') as f:
            pickle.dump(max_values, f)
        
        print("Loading Mask")
        all_masks = []
        for year in range(2018,2022):
            print(f'Mask {year}')
            filename = f'Mask_Mean_PM2.5_CONUS_{year}.mat'
            file = mat73.loadmat(os.path.join(self.data_path,filename))
            data = file[f'Mask_PM_STNs_{year}']
            data  = np.einsum('ijk->kij', data)

            for day in data:
                all_masks.append(day)
        self.mask = np.array(all_masks)
        
    
    def normalize(self, X):
        X = X.astype(np.float32)
        data = (X - np.min(X)) / (np.max(X) - np.min(X))
        return data
    
    def Random_Station_Mask(self, mask):
        
        # Create a random number
        np.random.seed(self.seed)
        
        num_sum = np.sum(mask) # Total number of pixels with stations inside
        
        try:
            num = np.random.randint(int(np.floor(num_sum/16)), high=int(np.floor(num_sum/8))) #12,4
        except:
            num = 1

        # # Create a copy a mask (named mask_2) and keep only part of stations locations (based on the random number)
        mask2 = np.argwhere(mask == 1)
        indices_1 = np.arange(mask2.shape[0])
        np.random.shuffle(indices_1)
        mask2 = mask2[indices_1]
        mask_new = np.zeros((mask.shape[0], mask.shape[1]))
        mask_new[mask2[num:, 0], mask2[num:, 1]] = 1 # We replace all missing pixels with 1, because this is how PCNN works
        
        return mask_new.astype(np.float32)

        
        
    def xr_aug(self):
        
        batch = self.num_of_samples
        img_size = self.img_size
        
        last_ind_lat = self.cmaq.shape[1] - img_size[0]
        last_ind_lon = self.cmaq.shape[2] - img_size[1]
        
        h = self.cmaq.shape[1]
        w = self.cmaq.shape[2]
        
        
        shift1 = np.append(np.arange(0,last_ind_lat, self.shift_var), last_ind_lat)
        shift2 = np.append(np.arange(0,last_ind_lon, self.shift_var), last_ind_lon)
        
        # CMAQ_indices
        indices_1 = np.arange(self.cmaq.shape[0]) #cmaq.shape[0]
        repeat = int(np.ceil((batch/self.cmaq.shape[0])))
        indices_1 = np.repeat(indices_1, repeat, axis=0)
        np.random.shuffle(indices_1)
        
        # Modis indices
        indices_2 = np.arange(self.modis.shape[0]) 
        repeat = int(np.ceil((batch/self.modis.shape[0])))
        indices_2 = np.repeat(indices_2, repeat, axis=0)
        np.random.shuffle(indices_2)
        
        # Mask indices
        indices_3 = np.arange(self.mask.shape[0]) 
        repeat = int(np.ceil((batch/self.mask.shape[0])))
        indices_3 = np.repeat(indices_3, repeat, axis=0)
        np.random.shuffle(indices_3)
        
        pus2 = np.zeros((batch, img_size[0], img_size[1], 1))
        modis2 = np.zeros((batch, img_size[0], img_size[1], 1))
        mask2 = np.ones((batch, img_size[0], img_size[1], 3))
        cmaq2 = np.zeros((batch, img_size[0], img_size[1], 1))

        # cmaq_norm = self.cmaq.max()
        # pus_norm = self.pus.max()
        # modis_norm = self.modis.max()
        cmaq_norm, pus_norm, modis_norm = 1,1,1
        
        for i in range(batch):
            if self.seed:
                np.random.seed(self.seed)
            else:
                self.seed = int(str(time.time()*1000000)[8:-2])
                np.random.seed(self.seed)
                
            # num = np.random.randint(1200, high=1500)/100 #110,130 / 10
            
            
            num1 = np.random.randint(h - self.img_size[0] + 1)
            num2 = np.random.randint(w - self.img_size[0] + 1)
        
            
            cmaq2[i,:,:] = self.cmaq[indices_1[i], num1:(num1+img_size[0]), \
                num2:num2+img_size[1], np.newaxis]/(cmaq_norm)
            
            
            modis2[i,:,:] = self.modis[indices_2[i], num1:(num1+img_size[0]), \
                num2:num2+img_size[1], np.newaxis]/(modis_norm)
            # modis2[modis2 < 0] = -0.1
            
            pus2[i,:,:] = self.pus[num1:(num1+img_size[0]), num2:num2+img_size[1], \
                np.newaxis]/(pus_norm)
            
            tmp = self.mask[indices_3[i], num1:(num1+img_size[0]), num2:num2+img_size[1]]
        
            mask2[i,:,:,0] = self.Random_Station_Mask(tmp)
            
        
        # cmaq2[mask2[:,:,:,0] == 0] = 1 # Only keep pixels in cmaq NO2 with stations inside
        print(np.nanmax(cmaq2), np.nanmax(modis2), np.nanmax(pus2))
        print(np.nanmin(cmaq2), np.nanmin(modis2), np.nanmin(pus2))
        final_input = np.concatenate((cmaq2, modis2, pus2), axis=3)
        final_mask = mask2
            
        return final_input.astype(np.float32), final_mask.astype(np.float32)  
        
    def __call__(self, number_of_samples, img_size, shift_var,  seed=123):
        
        self.num_of_samples = number_of_samples
        self.img_size = img_size
        self.shift_var = shift_var
        self.seed = seed
        print(f"Augumenting with seed {self.seed}")
        return self.xr_aug()


if __name__ == "__main__":
    data_path = "/tng4/users/skayasth/Yearly/2023/Jan/TCEQ/Data_for_PCNN"
    create_data = Data_prep(data_path)
    # second 768, 1024
    
    # las 896, 1536
    input_data, input_mask = create_data(2000, (768,1024), 12)
    input_data = np.einsum('nijk->nkij', input_data)
    input_mask = np.einsum('nijk->nkij', input_mask)
    np.save("data/input_data_2.npy", input_data)
    np.save("data/input_mask_2.npy", input_mask)

768, 1456
300 cmaq
130-140 modis
1 pus 

