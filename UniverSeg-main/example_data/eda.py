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
#dicom_data = tframe_data['dicom'][0][45] es differnt
dicom_data = tframe_data['dicom'][0][45] #el 45 es la medida con la que hay que relacionar el coordintes

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
coordinates_iwp = output_data['output']['Inner_with_Papillary_Muscles'][0, 45]
coordinates_iwop = output_data['output']['Inner_without_Papillary_Muscles'][0, 45]
coordinates_out = output_data['output']['Outer'][0, 45]
# Access the nested elements inside coordinates
# Access the nested elements inside coordinates
for nested_coordinates_iwp, nested_coordinates_iwop, nested_coordinates_out in zip(coordinates_iwp[0], coordinates_iwop[0], coordinates_out[0]):
    print("Shape of nested coordinates:", nested_coordinates_iwp.shape)
    # You can use the nested coordinates here for further processing

    # Scale the nested coordinates to match the dimensions of the input image
    scaled_coordinates_iwp = nested_coordinates_iwp * np.array([image_width / 255, image_height / 255])
    scaled_coordinates_iwop = nested_coordinates_iwop * np.array([image_width / 255, image_height / 255])
    scaled_coordinates_out = nested_coordinates_out * np.array([image_width / 250, image_height / 250]) #si aumento weight se va a la inziquiera
    # Plot the nested coordinates on top of the input image
    plt.imshow(input_image, cmap='gray')
    plt.plot(scaled_coordinates_iwp[:, 0], scaled_coordinates_iwp[:, 1], c='red', marker='.',
                label='Inner with Papillary Muscles')
    plt.plot(scaled_coordinates_iwop[:, 0], scaled_coordinates_iwop[:, 1], c='blue', marker='.',
                label='Inner without Papillary Muscles')
    plt.plot(scaled_coordinates_out[:, 0], scaled_coordinates_out[:, 1], c='green', marker='.',
                label='Outer contour')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Image with Coordinates')
    plt.legend()

    # Show the plot
    plt.show()