from aicspylibczi import CziFile
import os
import torch
import numpy as np

# Authored by Johannes Losert
# CZI type Dataset object for loading in czi files as tensors.
# supports any dimensionality (2d, 3d, video, image)
# transform can be used for preproeessing step (i.e. cropping, filters)
# 
# Important Note: if you are having trouble using this dataset, run "bash setup_paths.sh"
# from the home directory of the CVProject folder 

class CZIDataset(torch.utils.data.Dataset):
    
    def __init__(self, folder, transform=None):
        self.files = [folder + '/' + filename for filename in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load CZI file
        img_path = self.files[idx]
        img = CziFile(img_path)
        dims = img.dims
        shape = img.size
        frames_data, shape = img.read_image()
        # Convert image data to a torch tensor and remove dims with single element
        data = torch.from_numpy(np.squeeze(frames_data).astype(np.int16))

        # Apply transform if any
        if self.transform:
            data = self.transform(data)

        return data, dims, shape

