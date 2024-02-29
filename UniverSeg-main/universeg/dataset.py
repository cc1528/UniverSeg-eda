import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Matlab3DDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        if self.split == 'train':
            file_path = os.path.join(self.data_dir, 'train.mat')
        elif self.split == 'test':
            file_path = os.path.join(self.data_dir, 'test.mat')
        elif self.split == 'val':
            file_path = os.path.join(self.data_dir, 'val.mat')
        else:
            raise ValueError("Invalid split. Must be one of 'train', 'test', or 'val'.")
        return [file_path]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mat_file = self.file_list[idx]
        mat_data = sio.loadmat(mat_file)
        # Assuming 'image' is the key for the 3D image data
        image = mat_data['image']

        if self.transform:
            image = self.transform(image)

        return image

# Example usage:
# Define transformations

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    # Add more transformations as needed
])

batch_size = 32


# Create dataset instances for train, test, and validation sets
data_dir = r"C:\Users\cinth\Documentos\ams\data_science\actual_thesis\codes\UniverSeg-main\UniverSeg-main\example_data"

train_dataset = Matlab3DDataset(data_dir, split='train', transform=transform)
test_dataset = Matlab3DDataset(data_dir, split='test', transform=transform)
val_dataset = Matlab3DDataset(data_dir, split='val', transform=transform)

# Create dataloaders for train, test, and validation sets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
