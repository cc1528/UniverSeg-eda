#this code explores structure of output data

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.image import imsave

# Path to the training directory
output_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/O1.mat"
data = sio.loadmat(output_file_path)

    # Load the output MATLAB file

if os.path.exists(output_file_path):
    output_data = sio.loadmat(output_file_path)

    # Access the 'output' key
    output = output_data['output']

    # Print the keys within the 'output' dictionary
    # for key in output:
    #     print(key)

    # Accessing the 'Inner_with_Papillary_Muscles' field
    inner_wp = output['Inner_with_Papillary_Muscles']
    #inner_with_papillary_muscles = inner_with_papillary_muscles.astype(float)  # Convert to float dtype
    # Accessing the inner array
    inner_array = inner_wp[0, 0]
    print(np.shape(inner_wp))
    print(np.shape(inner_array))
    # Print the inner array
    #print(inner_array)
    # Access the innermost array within inner_array
    innermost_array = inner_array[0, 0]

    # Print the shape of the innermost array
    print("Shape of innermost array:", np.shape(innermost_array))

    # Print the innermost array
    #print(innermost_array)

    # Iterate over the rows of innermost_array
    # for row in innermost_array:
    #     print(row)

    # Access specific elements using indexing
    print("First element:", innermost_array[0])
    print("Second element:", innermost_array[1])
