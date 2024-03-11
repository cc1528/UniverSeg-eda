import os
import numpy as np
import scipy.io as sio
from pydicom import dcmread
import matplotlib.pyplot as plt
import re
from matplotlib.pyplot import imsave
from pydicom import dcmread

def load_coordinates(output_file_path, index):
    try:
        output_data = sio.loadmat(output_file_path)
        coordinates_iwp = output_data['output']['Inner_with_Papillary_Muscles'][0, index]
        coordinates_iwop = output_data['output']['Inner_without_Papillary_Muscles'][0, index]
        coordinates_out = output_data['output']['Outer'][0, index]
        return coordinates_iwp, coordinates_iwop, coordinates_out
    except IndexError:
        return None, None, None


def select_areas(input_image, scaled_coordinates_iwp, scaled_coordinates_iwop, scaled_coordinates_out):
    # Create masks for each area
    area_inside_iwp = np.zeros_like(input_image, dtype=bool)
    area_inside_iwp[scaled_coordinates_iwp[:, 1].astype(int), scaled_coordinates_iwp[:, 0].astype(int)] = True

    area_between_iwop_and_iwp = np.zeros_like(input_image, dtype=bool)
    area_between_iwop_and_iwp[scaled_coordinates_iwop[:, 1].astype(int), scaled_coordinates_iwop[:, 0].astype(int)] = True
    area_between_iwop_and_iwp[area_inside_iwp] = False

    area_between_out_and_iwop = np.zeros_like(input_image, dtype=bool)
    area_between_out_and_iwop[scaled_coordinates_out[:, 1].astype(int), scaled_coordinates_out[:, 0].astype(int)] = True
    area_between_out_and_iwop[area_between_iwop_and_iwp] = False

    # Select areas based on the masks
    return area_inside_iwp, area_between_iwop_and_iwp, area_between_out_and_iwop


def plot_image_with_coordinates(input_image, coordinates_iwp, coordinates_iwop, coordinates_out, image_width,
                                image_height, save_path, filename, index):
    if coordinates_iwp is None or coordinates_iwop is None or coordinates_out is None:
        return  # Skip if coordinates are None

    scaled_coordinates_iwp = coordinates_iwp * np.array([image_width / 255, image_height / 255])
    scaled_coordinates_iwop = coordinates_iwop * np.array([image_width / 255, image_height / 255])
    scaled_coordinates_out = coordinates_out * np.array([image_width / 255, image_height / 255])

    plt.imshow(input_image, cmap='gray')
    plt.plot(scaled_coordinates_iwp[:, 0], scaled_coordinates_iwp[:, 1], c='red', marker='.',
             label='Inner with Papillary Muscles')
    plt.plot(scaled_coordinates_iwop[:, 0], scaled_coordinates_iwop[:, 1], c='blue', marker='.',
             label='Inner without Papillary Muscles')
    plt.plot(scaled_coordinates_out[:, 0], scaled_coordinates_out[:, 1], c='green', marker='.',
             label='Outer contour')

    area_inside_iwp, area_between_iwop_and_iwp, area_between_out_and_iwop = select_areas(input_image,
                                                                                         scaled_coordinates_iwp,
                                                                                         scaled_coordinates_iwop,
                                                                                         scaled_coordinates_out)



    # Overlay all masks onto the input image
    plt.imshow(area_inside_iwp, cmap='Reds', alpha=0.7)
    plt.imshow(area_between_iwop_and_iwp, cmap='Greens', alpha=0.5)
    plt.imshow(area_between_out_and_iwop, cmap='Blues', alpha=0.3)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Image with Coordinates - ' + filename + '_' + str(index))
    plt.legend()

    plt.imsave(os.path.join(save_path, filename + '_' + str(index) + '.png'), input_image)

    plt.close()

input_folder_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/"
output_folder_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/results_3"

if os.path.exists(output_folder_path):
    for filename in os.listdir(output_folder_path):
        os.remove(os.path.join(output_folder_path, filename))
else:
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if re.match(r"I\d+\.mat", filename):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(input_folder_path, "O" + filename[1:])
        data = sio.loadmat(input_file_path)
        I_header = data['I']
        tframe_data = I_header['tframe'][0][0]

        file_number = filename[1:-4]

        # for j, dicom_file_path in enumerate(tframe_data['dicom'][0], start=0):
        #     ds = dcmread(dicom_file_path, force=True)
        #
        #     input_image = ds.pixel_array.astype(np.uint8)
        #
        #     image_height, image_width = input_image.shape[:2]
        #
        #     coordinates_iwp, coordinates_iwop, coordinates_out = load_coordinates(output_file_path, j)
        #     plot_image_with_coordinates(input_image, coordinates_iwp, coordinates_iwop, coordinates_out, image_width,
        #                                 image_height, output_folder_path, 'image_I' + file_number, j)


        # for j, dicom_data in enumerate(tframe_data['dicom'][0], start=0):
        #     dicom_data_uint8 = dicom_data.astype(np.uint8)
        #     dicom_file_path = os.path.join(input_folder_path, f"dicom_image_{file_number}_{j}.dcm")
        #     with open(dicom_file_path, "wb") as f:
        #         f.write(dicom_data_uint8.tobytes())
        #
        #     ds = dcmread(dicom_file_path, force=True)
        #
        #     input_image = ds.pixel_array.astype(np.uint8)
        #
        #     image_height, image_width = input_image.shape[:2]

        for j, dicom_data in enumerate(tframe_data['dicom'][0], start=0):
            imsave('image.png', dicom_data, cmap='gray')
            # Load the input image
            input_image_path = 'image.png'
            #input_image = imread(input_image_path)
            ds = dcmread(input_image_path, force = True)
            input_image = ds.pixel_array
            image_height, image_width = input_image.shape[:2]

            coordinates_iwp, coordinates_iwop, coordinates_out = load_coordinates(output_file_path, j)
            plot_image_with_coordinates(input_image, coordinates_iwp, coordinates_iwop, coordinates_out, image_width,
                                        image_height, output_folder_path, 'image_I' + file_number, j)
