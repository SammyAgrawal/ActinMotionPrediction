from aicspylibczi import CziFile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.animation as animation
import IPython
from IPython.display import HTML
import ffmpeg

from scipy.stats import sigmaclip
from multiprocessing import Pool
import os
import cv2
from skimage.measure import label, regionprops_table # version at least 0.22
from skimage.filters import threshold_otsu # version at least 0.22


def get_file(vid_type, num):
    root_dir = "/mnt/datadisk/"
    vid_type = vid_type.lower()
    if(vid_type == "processed"):
        folder = 'FactinProcessed'
        fname = f'dicty_factin_pip3-0{num}_processed.czi'
    elif(vid_type == 'mip'):
        folder = 'FactinMIP'
        fname = f'dicty_factin_pip3-0{num}_MIP.czi'
    elif(vid_type == 'new'):
        folder = 'FactinMIP'
        fname = f'New-0{num}_MIP.czi'
    else:
        print(f"invalid type, try again from {['processed', 'mip', 'new']}")
        return(-1)
    fpath = os.path.join(root_dir, folder, fname)
    video = CziFile(fpath)
    print(f"Loading {fname} with dims {video.get_dims_shape()}")
    return(video)


def scale_img(img, do_clip=True):
    lower_bound = np.percentile(img, 1)
    upper_bound = np.percentile(img, 99.9)
    I = (img - lower_bound) / (upper_bound - lower_bound)
    if(do_clip):
        I = np.clip(I, 0, 1)
    return(I)


def binarize_video(frames, thresh_calc='all'):
    binary_frames = []
    if (thresh_calc=='all'):
        for frame_idx, frame in enumerate(frames):
            thresh=threshold_otsu(frame)
            B = frame >= thresh
            B = label(B, background=0, connectivity=2)
            binary_frames.append(B)
            
    elif (thresh_calc == 'first'):
        # use threshhold in first frame across entire video
        thresh = threshold_otsu(frames[0])
        for frame_idx, frame in enumerate(frames):
            B = frames[frame_idx] >= thresh
            B = label(B, background=0, connectivity=2)
            binary_frames.append(B)
    
    ## func for labelling connected components
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

    return(np.array(binary_frames))

def frame_otsu_thresholding(frame):
    return (frame >= threshold_otsu(frame))

def frame_connected_components(thresholded_frame):
    return label(thresholded_frame, background=0, connectivity=2)

def binarize_video_fast(frames):

    # computes a unique otsu threshhold for all videos
    with Pool(processes=os.cpu_count()) as pool:
        # Map apply_otsu_threshold to each frame
        video_threshed = pool.map(frame_otsu_thresholding, frames)
        video_labeled = pool.map(frame_connected_components, video_threshed)
    
    return video_labeled
    

def bounding_boxes(mask, min_area=300):    
    # deducing centroids, max side length, 
    all_bboxes_coords = list()
    max_num_cells=0
    num_cells_frames = list()
    # region props can calculate a LOT of useful properties
    # might want to look into better cell filtering
    props = regionprops_table(label_image=mask,
                              properties=['bbox','area']
    )
    # each row corresponds to cell id, columns corresponds to properties
    # box returned as min_row, min_col, max_row, max_col
    df = pd.DataFrame(props)
    df = df[df['area'] >= min_area]
    cell_areas = df['area'].values
    df= df.drop(columns = ['area'])
    bboxes, num_cells = list(df.itertuples(index=False, name=None)), df.shape[0]
    return bboxes, num_cells, cell_areas


def draw(img, box, thickness, val):
    min_row, min_col, max_row, max_col = box
    img = cv2.rectangle(img, (min_col, min_row), (max_col, max_row), val, thickness)
    return(img)

def draw_boxes(frame, boxes, thickness=2, val=1, cell_id=-1):
    img = frame.copy()
    
    if(cell_id==-1):
        for bbox in boxes:
            img = draw(img, bbox, thickness, val)
    else:
        img = draw(img, boxes[cell], thickness, val) 
        
    return(img)



def box_tracking_video(frames, masks=-1, thickness=2):
    if(type(masks) == int):
        # if not provided, calculated
        masks = binarize_video(frames)
        print("Computed binary masks")
    video = []
    num_cells_per_frame = []
    cell_areas = []
    for frame_idx in range(len(frames)):
        boxes, num_cells, areas = bounding_boxes(masks[frame_idx])
        num_cells_per_frame.append(num_cells)
        cell_areas.append(areas)

        img_with_boxes = draw_boxes(frames[frame_idx], boxes, thickness, val=1)
        video.append(img_with_boxes)
        
    return(video, num_cells_per_frame, cell_areas)


def visualize_cell_tracker(frames, data, thickness=2, val=1):
    video_frames = []
    for frame_idx in range(len(frames)):
        box = data['boxes'][frame_idx]
        img = draw(frames[frame_idx].copy(), box, thickness, val)
        video_frames.append(img)
    return(video_frames)


def animate_frames(frames):
    fig = plt.figure()
    im = plt.imshow(frames[0])
    
    plt.close() # this is required to not display the generated image
    
    def init():
        im.set_data(frames[0])
    
    def animate(i):
        im.set_data(frames[i])
        return im
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames),
                                   interval=200)
    
    HTML(anim.to_html5_video())
    plt.show()
    return(anim)

def extract_traces(frames, masks, hist=2, load_verifying_videos=True):
    bboxes, num_cells, areas = bounding_boxes(masks[0])
    vid_data = []
    videos_for_checking = []
    for i in range(num_cells):
        print("Extracting cell ", i)
        data = track_cells(i, frames, masks, padding=0, history_length=hist, verbose=False)
        vid_data.append(data)
        if(load_verifying_videos):
            videos_for_checking.append(np.array(visualize_cell_tracker(frames, data)))

    return(vid_data, videos_for_checking)


def track_cells(cell_id, frames, masks=-1, padding=0, history_length=1, verbose=False):
    N = frames.shape[0]
    if(type(masks) == int):
        # if not provided, calculated
        masks = binarize_video(frames)
        print("Computed binary masks")

    data = {"patches" : [], 'boxes' : [], "masks" : []}
    areas_frames = []
    num_cells_frames = []
    
    def get_corresponding_cell_box(curr_boxes, curr_num, curr_areas, frame_idx, num_past_to_consider, verbose=False):
        ## TO DO : Ensure that index of boxes stays consistent even as boxes appear and dissapear
        if(frame_idx == 0):
            return(cell_id)

        assert len(data['boxes']) >= frame_idx, f"Missing frame history at {frame_idx}"
        
        scores = np.zeros(len(boxes))
        for i in range(max(0, len(data['boxes']) - num_past_to_consider), len(data['boxes'])):
            prev_box, prev_num, prev_area = data['boxes'][i], num_cells_frames[i], areas_frames[i]
            for j in range(len(scores)):
                delta_area = abs(prev_area - curr_areas[j])
                delta_x = abs(prev_box[0] - curr_boxes[j][0])
                delta_y = abs(prev_box[1] - curr_boxes[j][1])
                if(verbose):
                    print(f"Score {j} comprised of {delta_area}, {delta_x}, {delta_y}")
                scores[j] += 50*(delta_x + delta_y) + delta_area / 30
        index = np.argmin(scores)
        if(index != cell_id and verbose):
            print(f"Choosing {index} instead of cell id {cell_id} at frame {frame_idx}\nAreas: {curr_areas}\n boxes: {curr_boxes}")
            print(f"Previous box and area: {prev_box}\n {prev_area}")
            print("\n\n\n")
        return(index)
    
    h_max, w_max = 0, 0
    skip_frames = []
    for frame_idx in range(N):
        boxes, num_cells, areas = bounding_boxes(masks[frame_idx]) # all on frame
        index_of_cell = get_corresponding_cell_box(boxes, num_cells, areas, frame_idx, history_length, verbose) # right now assume same index throughout entire video
        
        bbox = boxes[index_of_cell] # cell id is meant to track a SPECIFIC cell, ensure maintain integrity of cell over time
        min_row, min_col, max_row, max_col = bbox
        #TODO might expand boxes to homogenize sizes; 
        min_row -= padding
        min_col -= padding # adjust boxes based on padding
        max_row += padding
        max_col += padding

        patch = frames[frame_idx, min_row:max_row, min_col:max_col]
        mask_patch = masks[frame_idx, min_row:max_row, min_col:max_col]

        data['boxes'].append((min_row, min_col, max_row, max_col))
        data['patches'].append(patch)
        data['masks'].append(mask_patch)

        num_cells_frames.append(num_cells)
        areas_frames.append(areas[index_of_cell])

    return(data)






