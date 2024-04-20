from aicspylibczi import CziFile
import os
import torch
import numpy as np
from skimage.measure import label # version at least 0.22
from skimage.measure import regionprops_table # version at least 0.22
from skimage.filters import threshold_otsu # version at least 0.22
import pandas as pd

#import torch_datasets.czi as czi
#


# Authored by Johannes Losert
# CZI type Dataset object for loading in czi files as tensors.
# supports any dimensionality (2d, 3d, video, image)
# transform can be used for preproeessing step (i.e. cropping, filters)
# 
# Important Note: if you are having trouble using this dataset, run "bash setup_paths.sh"
# from the home directory of the CVProject folder 

class CZIDataset(torch.utils.data.Dataset):
    
    def __init__(self, folder, transform=None, twod_vid_channel=None):
        self.files = [folder + '/' + filename for filename in os.listdir(folder)]
        self.transform = transform
        self.twod_vid_channel = twod_vid_channel

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
        data = np.squeeze(frames_data).astype(np.uint32)

        # Apply transform if any
        if self.transform:
            data = self.transform(data)
        if (type(self.twod_vid_channel) == int and self.twod_vid_channel >= 0 \
            and len(data.shape)==4):
            data = np.squeeze(data[:,self.twod_vid_channel,:,:]) 

        return data, dims, shape

#TODO_1: return a numpy ndarray instead of a list by adding padding
#TODO_2: perform cell matching across frames using bounding box similarity
#TODO_3: optimize memory usage by minimizing internmediate objects / being more pythonic
#       / using kernel fusion


class Transforms2D:

    # borrowed from sammy. can be passed as a "transform argument"
    # since our intensities can get very large, we want to clip exploding values 
    # and bring all the values into the range(0,1)
    @staticmethod
    def clip_intensities(video):
        lower_bound, upper_bound = np.percentile(img, [1, 99.9])
        img = (img - lower_bound) / (upper_bound - lower_bound)
        np.clip(video, 0, 1, out=img)  # in-place clipping
        return video
    
    @staticmethod
    def thresh(video):
        threshes = [threshold_otsu(frame) for frame in video]
        return np.array([frame >= thresh for frame, thresh in zip(video, threshes)]) 
    
    # returns a 'labeled' frame
    @staticmethod
    def connected_components(thresholded_frame):
         return label(thresholded_frame,background=0,connectivity=2)
    
    @staticmethod
    def bounding_boxes(labeled_frame, min_area=200):
        props = regionprops_table(labeled_frame, properties=['bbox', 'area'])
        df = pd.DataFrame(props)
        
        # Filter the DataFrame for areas greater than min_area
        filtered_df = df[df['area'] >= min_area]
    
        # Extract the bounding box coordinates
        bbox_coords = [
            (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'])
            for index, row in filtered_df.iterrows()
        ]
        
        return bbox_coords

    @staticmethod
    def coords_to_bbox_frame_list(frame, bbox_coords):
        return  [frame[min_row:max_row, min_col:max_col] for min_row, min_col, max_row, max_col in bbox_coords]
        
    
    @staticmethod
    def default_bounding_boxes_pipeline_2dvideo(video):
        video = np.array([Transforms2D.clip_intensities(frame) for frame in video])
        thresholded_video = np.array([Transforms2D.thresh(frame) for frame in video])
        labeled_video = np.array([Transforms2D.connected_components(thresholded_frame) for thresholded_frame in thresholded_video])
        bounding_boxes = [Transforms2D.bounding_boxes(labeled_frame) for labeled_frame in labeled_video]
        bounding_box_frames = [Transforms2D.coords_to_bbox_frame_list(frame,bbox_coords) for frame, bbox_coords in zip(video, bounding_boxes)]
        return bounding_box_frames
    