from aicspylibczi import CziFile
import os
import torch
import numpy as np
from skimage.measure import label # version at least 0.22
from skimage.measure import regionprops_table # version at least 0.22
from skimage.filters import threshold_otsu # version at least 0.22
import pandas as pd
from scipy.stats import sigmaclip
from multiprocessing import Pool
import os

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
        data = np.squeeze(frames_data).astype(np.uint16)

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


    # midpoint is super efficient because integer datatypes means bitlevel right shift of difference 
    @staticmethod
    def clip_intensities(video):
        percentile_axes = None
        if (video.ndim == 4) : percentile_axes = (1,2,3)
        elif (video.ndim == 3): percentile_axes = (1,2)
        
        frames_lower_bounds, frames_upper_bounds = np.percentile(video, [1, 99.9], axis=percentile_axes,method='midpoint')
        
        if (video.ndim == 4):        
            frames_lower_bounds = frames_lower_bounds[:, np.newaxis, np.newaxis, np.newaxis]
            frames_upper_bounds = frames_upper_bounds[:, np.newaxis, np.newaxis, np.newaxis]
        elif (video.ndim == 3):
            frames_lower_bounds = frames_lower_bounds[:, np.newaxis, np.newaxis]
            frames_upper_bounds = frames_upper_bounds[:, np.newaxis, np.newaxis]
            
        video = (video - frames_lower_bounds) / (frames_upper_bounds - frames_lower_bounds)
        np.clip(video, 0, 255, out=video).astype(np.uint8)
        return video

    # computes the otsu tresholded version of a single frame
    # only works for 2d
    # need to make it work for 3d or find diff segmentation
    @staticmethod
    def frame_otsu_thresholding(frame):
        return (frame >= threshold_otsu(frame))

    # computes a unique otsu threshhold for all videos
    @staticmethod
    def video_otsu_thresholding_parallel(video):
        with Pool(processes=os.cpu_count()) as pool:
            # Map apply_otsu_threshold to each frame
            video_threshed = pool.map(Transforms2D.frame_otsu_thresholding, video)
        return video_threshed
    
    @staticmethod
    def frame_connected_components(thresholded_frame):
        return label(thresholded_frame,background=0,connectivity=2)
    
    @staticmethod
    def video_connected_components_parallel(thresholded_video):
        with Pool(processes=os.cpu_count()) as pool:
            video_labeled = pool.map(Transforms2D.frame_connected_components, thresholded_video)
        return video_labeled
    

    @staticmethod
    def frame_bounding_boxes(labeled_frame, min_area=250):
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
    def video_bounding_boxes_parallel(labeled_video):
        with Pool(processes=os.cpu_count()) as pool:
            video_labeled = pool.map(Transforms2D.frame_bounding_boxes, labeled_video)
        return video_labeled

    @staticmethod
    def apply_bounding_boxes (frame, bbox_coords):
        return  [frame[min_row:max_row, min_col:max_col] for min_row, min_col, max_row, max_col in bbox_coords]
        