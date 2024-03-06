import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave

# Load the input image
input_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/I1.mat"
data = sio.loadmat(input_file_path)
I_header = data['I']
tframe_data = I_header['tframe'][0][0]
dicom_data = tframe_data['dicom'][0][0]

# Save the image to a file
imsave('image.png', dicom_data, cmap='gray')  # Assuming grayscale image, change cmap if needed

# Load the input image
input_image_path = 'image.png'
input_image = imread(input_image_path)

# Get the dimensions of the input image
image_height, image_width = input_image.shape[:2]

# Load the coordinates from the output file
output_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/O1.mat"
output_data = sio.loadmat(output_file_path)
coordinates = output_data['output']['Inner_with_Papillary_Muscles'][0, 0]
#print(coordinates)

# # Scale the coordinates to match the dimensions of the input image
#scaled_coordinates = np.array(coordinates) * np.array([image_width / 339, image_height / 413])
#print(scaled_coordinates)

# Scale the coordinates to match the dimensions of the input image
-
# x_coordinates = scaled_coordinates[:, 0]
# y_coordinates = scaled_coordinates[:, 1]
#
# # Plot the input image
# plt.imshow(input_image, cmap='gray')
#
# # Plot the coordinates on top of the input image
# plt.scatter(x_coordinates, y_coordinates, c='red', marker='.', label='Coordinates')
#
# # Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Input Image with Coordinates')
plt.legend()

# Show the plot
plt.show()