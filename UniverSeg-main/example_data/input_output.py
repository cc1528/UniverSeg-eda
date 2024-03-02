import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Path to the training directory
input_image_path = "/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/I1.mat"
output_mask_path = "/mnt/c/Users/cinth/Documentos/ams/data_science/actual_thesis/codes/UniverSeg-main/UniverSeg-main/example_data/Training/O1.mat"

data = sio.loadmat(input_image_path)

# Access the 'I' header
I_header = data['I']

# Check if it's a numpy ndarray
if isinstance(I_header, np.ndarray):
    # Access the 'tframe' field
    tframe_data = I_header['tframe'][0][0]

    # Access the 'dicom' field within 'tframe'
    input_image = tframe_data['dicom'][0][0]

if os.path.exists(output_mask_path):
    output_data = sio.loadmat(output_mask_path)

    # Access the 'output' key
    output = output_data['output']

    inner_wp = output['Inner_with_Papillary_Muscles']

    output_mask = inner_wp[0, 0]
# Plot the input image
plt.imshow(input_image, cmap='gray')

# Overlay the mask on top of the input image
plt.imshow(output_mask, alpha=0.5, cmap='jet')  # Adjust alpha and colormap as needed
plt.colorbar()  # Add colorbar for reference if needed

plt.title('Input Image with Overlayed Mask')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
