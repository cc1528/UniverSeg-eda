import os
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import scipy.io as sio

# print("Starting script...")
#
# class Matlab3DDataset:
#     def __init__(self, data_dir, split='train'):
#         self.data_dir = data_dir
#         self.split = split
#         self.input_files, self.output_files = self._get_file_lists()
#
#     def _get_file_lists(self):
#         input_files = []
#         output_files = []
#
#         if self.split == 'train':
#             folder = 'Training'
#             num_files = 16
#         elif self.split == 'test':
#             folder = 'Test'
#             num_files = 5
#         elif self.split == 'val':
#             folder = 'Validation'
#             num_files = 5
#         else:
#             raise ValueError("Invalid split. Must be one of 'train', 'test', or 'val'.")
#
#         data_folder = os.path.join(self.data_dir, folder)
#
#         for i in range(1, num_files + 1):
#             input_file = os.path.join(data_folder, f"I{i}.mat")
#             output_file = os.path.join(data_folder, f"O{i}.mat")
#
#             if os.path.exists(input_file) and os.path.exists(output_file):
#                 input_files.append(input_file)
#                 output_files.append(output_file)
#
#         return input_files, output_files
#
#     def compute_image_dimensions_statistics(self):
#         image_dimensions = []
#
#         for input_file, output_file in zip(self.input_files, self.output_files):
#             input_data = sio.loadmat(input_file)
#             output_data = sio.loadmat(output_file)
#
#             input_image_shape = input_data.get('tframe', None).shape
#             output_image_shape = output_data.get('Inner_with_Papillary_Muscles', None).shape
#
#             image_dimensions.append(input_image_shape)
#             image_dimensions.append(output_image_shape)
#
#         # Compute statistics
#         min_dims = min(image_dimensions)
#         max_dims = max(image_dimensions)
#         avg_dims = sum(image_dimensions) / len(image_dimensions)
#
#         print(f"Minimum image dimensions across {self.split} images:", min_dims)
#         print(f"Maximum image dimensions across {self.split} images:", max_dims)
#         print(f"Average image dimensions across {self.split} images:", avg_dims)
#
#     def __len__(self):
#         return len(self.input_files)
#
#
# # Create dataset instances for train, test, and validation sets
# data_dir = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data"
# print("Data directory:", data_dir)
#
# print("Data directory exists:", os.path.exists(data_dir))
# train_dataset = Matlab3DDataset(data_dir, split='train')
# test_dataset = Matlab3DDataset(data_dir, split='test')
# val_dataset = Matlab3DDataset(data_dir, split='val')
#
# # Compute image dimension statistics for each dataset
# train_dataset.compute_image_dimensions_statistics()
# test_dataset.compute_image_dimensions_statistics()
# val_dataset.compute_image_dimensions_statistics()


import scipy.io as sio
import matplotlib.pyplot as plt

# Load the .mat file
input_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/I1.mat"
data = sio.loadmat(input_file_path)

# # Iterate through keys and examine their contents
# for key in data.keys():
#     print("Contents of key '{}':".format(key))
#     if isinstance(data[key], np.ndarray):
#         print("Shape:", data[key].shape)
#         # If the data is an array, you can plot it
#         if data[key].ndim == 2:
#             plt.imshow(data[key], cmap='gray')  # assuming grayscale image
#             plt.title(key)
#             plt.colorbar()
#             plt.show()
#     else:
#         print(data[key])

# Perform additional analysis and visualizations as needed

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio

# Load the MATLAB file
import scipy.io as sio

# Load the MATLAB file
import scipy.io as sio
import numpy as np

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# Load the MATLAB file
import scipy.io as sio
import numpy as np

# Load the MATLAB file
import scipy.io as sio
import numpy as np

# Load the MATLAB file
input_file_path = r"/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/I1.mat"
data = sio.loadmat(input_file_path)

# Access the 'I' header
I_header = data['I']

# Check if it's a numpy ndarray
if isinstance(I_header, np.ndarray):
    # Print basic statistical summary of the 'I' header array
    print("Shape of the 'I' header array:", I_header.shape)
    print("Data type of the 'I' header array:", I_header.dtype)

    # Accessing the 'fstart' field and computing its minimum value
    fstart_min = np.min(I_header['fstart'][0][0])
    print("Minimum value of 'fstart':", fstart_min)

    # Display the ndarray
    # print("Contents of the 'I' header:")
    # print(I_header)
else:
    print("The 'I' header is not a numpy ndarray.")
