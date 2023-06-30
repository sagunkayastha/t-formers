import os
import ast
import glob
import mat73
import pylab
import pickle
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt


# Set up configuration
img_shape = (512, 512)
test_model = "/tng4/users/skayasth/Yearly/2023/Jan/TCEQ/Modified_PCNN/runs/512_tf/512_tf.h5"
output_folder = "outputs"
test_shape = (896, 1536)
data_path = "/tng4/users/skayasth/Yearly/2023/Jan/TCEQ/Data_for_PCNN"
max_file = "/tng4/users/skayasth/Yearly/2023/Jan/TCEQ/Modified_PCNN/runs/512_old_arch/maxes.pkl"

# Save dictionary to a file
with open('Station_dict.pkl', 'rb') as file:
    Station_dict = pickle.load(file)
    
with open('Mask_dict.pkl', 'rb') as file:
    Mask_dict = pickle.dump(file)
    
    
# Predict
Station_Output = {}
for year, _ in Station_dict.items():
    input_data_all = Station_dict[year]
    input_mask_all = Mask_dict[year]

    output_daily = np.zeros(
        (input_data_all.shape[0], input_data_all.shape[1], input_data_all.shape[2]))

    # Loop through each day
    # Phases 1 . Predicting the output for each day with all the stations
    for day, (input_data, input_mask) in enumerate(zip(input_data_all, input_mask_all)):

        # Do Noraml imputation without removing the station (Similar to the previous method)
        # padding input and mask for making image size 896x1536

        
        # Define the desired padded dimensions
        original_array = input_data

        padded_height, padded_width = 1024, 1536


        # Get the original array shape
        original_height, original_width, original_channels = original_array.shape

        # Calculate the required amount of padding
        pad_height = padded_height - original_height
        pad_width = padded_width - original_width

        # Calculate the padding sizes for each dimension
        top_padding = pad_height // 2
        bottom_padding = pad_height - top_padding
        left_padding = pad_width // 2
        right_padding = pad_width - left_padding


        # Perform the zero-padding
        day_in_padded = np.pad(input_data, ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), mode='constant')
        mask_in_padded = np.pad(input_mask, ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), mode='constant')
        

        # unpadded_array = padded_array[top_padding:top_padding+original_height, left_padding:left_padding+original_width, :]


        # Expand the dimensions to make it compatible with the model
        i, j = np.expand_dims(day_in_padded, axis=0), np.expand_dims(
            mask_in_padded, axis=0)

        # Predict the output and multiply by 450 to get the original values
        daily_out = model.predict([i, j]) * 450.

        # Only take the output for the original image size
        output_daily[day, :, :] = daily_out[0, 59:837, 40:1496, 0]

        # For Phase 2 imputation
        # Get the indices of the stations
        one_indices_mask = np.where(input_mask[:, :, 0] == 1)
        index_pairs_mask = np.column_stack(one_indices_mask)
        output_station = np.zeros(
            (len(index_pairs_mask), input_data_all.shape[1], input_data_all.shape[2]))
        print(
            f'Imputing year: {year}, day: {day}, with {len(index_pairs_mask)} stations')

        # Loop through each station for Phase 2
        for n, ip in enumerate(index_pairs_mask):

            # Change mask to 0 for the station (ie impute the station)
            input_mask[ip[0], ip[1], 0] = 0
            input_data[ip[0], ip[1], 0] = 0

            # Pad data and mask to make input size 896x1536
            day_in_padded = np.zeros((896, 1536, 3))
            day_in_padded[59:837, 40:1496, :] = input_data
            mask_in_padded = np.ones((896, 1536, 3))
            mask_in_padded[59:837, 40:1496, :] = input_mask

            # Expand the dimensions to make it compatible with the model
            i, j = np.expand_dims(day_in_padded, axis=0), np.expand_dims(
                mask_in_padded, axis=0)

            station_out = model.predict([i, j]) * 450.
            output_p2 = station_out[0, 59:837, 40:1496, 0]

            output_daily[day, ip[0], ip[1]
                         ] = output_p2[ip[0], ip[1]]

    Station_Output[year] = output_daily
