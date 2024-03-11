
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import re

def load_coordinates(output_file_path, index):
    try:
        output_data = sio.loadmat(output_file_path)
        coordinates_iwp = output_data['output']['Inner_with_Papillary_Muscles'][0, index]
        coordinates_iwop = output_data['output']['Inner_without_Papillary_Muscles'][0, index]
        coordinates_out = output_data['output']['Outer'][0, index]
        return coordinates_iwp, coordinates_iwop, coordinates_out
    except IndexError:
        return None, None, None


def plot_image_with_coordinates(input_image, coordinates_iwp, coordinates_iwop, coordinates_out, image_width,
                                image_height, save_path, filename, index):
    if coordinates_iwp is None or coordinates_iwop is None or coordinates_out is None:
        return  # Skip if coordinates are None

    for nested_coordinates_iwp, nested_coordinates_iwop, nested_coordinates_out in zip(coordinates_iwp[0],
                                                                                       coordinates_iwop[0],
                                                                                       coordinates_out[0]):
        scaled_coordinates_iwp = nested_coordinates_iwp * np.array([image_width / 255, image_height / 255])
        scaled_coordinates_iwop = nested_coordinates_iwop * np.array([image_width / 255, image_height / 255])
        scaled_coordinates_out = nested_coordinates_out * np.array([image_width / 255, image_height / 255])

        plt.imshow(input_image, cmap='gray')
        plt.plot(scaled_coordinates_iwp[:, 0], scaled_coordinates_iwp[:, 1], c='red', marker='.',
                 label='Inner with Papillary Muscles')
        plt.plot(scaled_coordinates_iwop[:, 0], scaled_coordinates_iwop[:, 1], c='blue', marker='.',
                 label='Inner without Papillary Muscles')
        plt.plot(scaled_coordinates_out[:, 0], scaled_coordinates_out[:, 1], c='green', marker='.',
                 label='Outer contour')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Input Image with Coordinates - ' + filename + '_' + str(index))
        plt.legend()
        plt.savefig(os.path.join(save_path, filename + '_' + str(index) + '.png'))
        plt.close()


# Input paths
input_folder_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/"
output_folder_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/results"

# Create the output folder if it doesn't exist and remove existing images
if os.path.exists(output_folder_path):
    for filename in os.listdir(output_folder_path):
        os.remove(os.path.join(output_folder_path, filename))
else:
    os.makedirs(output_folder_path)


# Loop over all files in the input folder
for filename in os.listdir(input_folder_path):
    if re.match(r"I\d+\.mat", filename):
        input_file_path = os.path.join(input_folder_path, filename)

        output_file_path = os.path.join(input_folder_path, "O" + filename[1:])  # Matching output file name
        data = sio.loadmat(input_file_path)
        I_header = data['I']
        tframe_data = I_header['tframe'][0][0]

        # Extract the number of the I file
        file_number = filename[1:-4]

        # Loop over all images in 'dicom' array
        for j, dicom_data in enumerate(tframe_data['dicom'][0], start=0):
            imsave('image.png', dicom_data, cmap='gray')
            # Load the input image
            input_image_path = 'image.png'
            input_image = imread(input_image_path)
            image_height, image_width = input_image.shape[:2]

            # Load the coordinates
            coordinates_iwp, coordinates_iwop, coordinates_out = load_coordinates(output_file_path, j)
            plot_image_with_coordinates(input_image, coordinates_iwp, coordinates_iwop, coordinates_out, image_width,
                                        image_height, output_folder_path, 'image_I' + file_number, j)
