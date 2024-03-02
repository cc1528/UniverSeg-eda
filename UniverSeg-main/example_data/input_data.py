import os
#this code releases images of the input images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from matplotlib.image import imsave

# Load the MATLAB file
input_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/I1.mat"
data = sio.loadmat(input_file_path)

# Access the 'I' header
I_header = data['I']

# Check if it's a numpy ndarray
if isinstance(I_header, np.ndarray):
    # Access the 'tframe' field
    tframe_data = I_header['tframe'][0][0]

    # Access the 'dicom' field within 'tframe'
    dicom_data = tframe_data['dicom'][0][0]

    # Save the image to a file
    imsave('image.png', dicom_data, cmap='gray')  # Assuming grayscale image, change cmap if needed
    print("Image saved as 'image.png'")
else:
    print("The 'I' header is not a numpy ndarray.")

