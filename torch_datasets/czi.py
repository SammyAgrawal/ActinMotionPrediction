from aicspylibczi import CziFile
import os
import torch
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu


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
        data = np.squeeze(frames_data).astype(np.int16)

        # Apply transform if any
        if self.transform:
            data = self.transform(data)

        data=torch.from_numpy(data)

        return data, dims, shape

#
class Transforms2D:

    # you may not want to do the whole video if too big
    def bounding_boxes(video,first_frame = 0,last_frame= -1):
        # background subtraction by threshholding based on the first frame
        first_frame = video[0]
        first_frame_thresh = threshold_otsu(first_frame)
        first_frame = None

        # labeling connected components of each frame
        video = video[first_frame:last_frame] >= first_frame_thresh
        labeled_video = np.array([label(frame) for frame in video])

        # array of dictionaries for bounding boxes
        all_bboxes = list()
        for frame in labeled_video:
            all_props = regionprops_table(frame, cache=True)
            all_bboxes.append(all_props['bbox'])
            
        # jagged list of size (frames, num bboxes in frame, intensity of each bbox)
        bboxes_video = list()
        for idx, frame in enumerate(video):
            bboxes = all_bboxes[idx]
            bboxes_frame = list()
            for bbox in bboxes:
                min_row, min_col, max_row, max_col = bbox
                bboxes_frame.append(frame[min_row:max_row, min_col:max_col])
            bboxes_video.append(bboxes_frame)
            
        
  
        
    