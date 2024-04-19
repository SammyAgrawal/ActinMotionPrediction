from aicspylibczi import CziFile
import os
import torch
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
import pandas as pd


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

    # returns the data in numpy format for easier preprocessing
    def __getitem__(self, idx):
        # Load CZI file
        img_path = self.files[idx]
        img = CziFile(img_path)
        dims = img.dims
        shape = img.size
        frames_data, shape = img.read_image()
        # Convert image data to a torch tensor and remove dims with single element
        data = np.squeeze(frames_data).astype(np.int32)

        # Apply transform if any
        if self.transform:
            data = self.transform(data)

        return data, dims, shape


class Transforms2D:

    # you may not want to do the whole video if too big
    def bounding_boxes(video,first_frame = 0,last_frame= -1, max_side_length=None):
        video = video[first_frame:last_frame]
        
        # background subtraction by threshholding based on the first frame
        thresh = np.average(np.array([threshold_otsu(frame) for frame in video]))

        # labeling connected components of each frame
        video = video >= thresh
        
        labeled_video = np.array([label(frame) for frame in video]) 

        # deducing centroids, max side length, 
        all_bboxes = list()
        max_num_cells = 0
        max_bbox_size = 0

        for labeled_frame in labeled_video:
            props = regionprops_table(label_image=labeled_frame,
                                      properties=['bbox']
            )
            df = pd.DataFrame(props)
            df['h'] = df['bbox-2'] - df['bbox-0'] 
            df['w'] =  df['bbox-3'] - df['bbox-1']
            max_side_length = max(max(df['h'], df['w']))


        # keep the bounding box size within the bounds
        if (type(max_side_length) == int and max_bbox_size > max_side_length):
            max_bbox_size = max_side_length

        # make the bounding box size even
        if (max_bbox_size % 2 == 1):
            max_bbox_size = max_bbox_size + 1
            
        # numpy ndarray of shape (frames, max_num_cells, max_bbox_size, max_bbox_size)
        frames=video.shape[0]
        bboxes_video = np.zeros((frames, max_num_cells, max_bbox_size, max_bbox_size))
        for frame_idx, frame in enumerate(video):
            for bbox_idx, bbox in enumerate(all_bbox[frame_idx]):
                min_row = max(0,max_bbox_size)
                max_row = min(0)
                height = max_row - min_row

                min_col = max(0,centroid[1] - max_bbox_size//2,0)
                max_col = min(frame.shape[1], centroid[1] + max_bbox_size//2)
                width = max_col - min_col

                bboxes_video[frame_idx,centroid_idx,0:height,0:width] = frame[min_row:max_row, min_col:max_col]
        
        return bboxes_video
        
  
        
    