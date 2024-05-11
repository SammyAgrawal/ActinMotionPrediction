from utils import *
import mahotas #: Module("mahotas")


def extract_traces_sparse(frames, masks, hist=2):
    bboxes, num_cells, areas = bounding_boxes(masks[0])
    vid_data = []
    for i in range(num_cells):
        #print("Extracting cell ", i)
        data = track_cells(i, frames, masks, padding=0, history_length=hist, verbose=False)
        vid_data.append(data)
    return(vid_data)

def shape_features(binary, feature_length=20, num_samples=180):
    def radial_distance(binary, theta):
        height, width = binary.shape
        center = [width // 2, height // 2]
        def test_r(r):
            x_test, y_test = center[0] + r*np.cos(theta), center[1] + r*np.sin(theta)
            if(x_test >= width or y_test > height or x_test < 0 or y_test < 0):
                return(False)
            return(binary[int(y_test), int(x_test)])
        # calculate distance to the nearest pixel
        r = max(height, width)
        while(not test_r(r)): # start from edge come inside until hit cell
            r -= 1
        return(r)

    test_angles = np.linspace(0, 2*np.pi, num_samples)
    distances = np.array([radial_distance(binary, angle) for angle in test_angles])
    fft_coefficients = np.fft.rfft(distances)

    features = np.abs(fft_coefficients[:feature_length])
    features = features / np.sum(features)
    return(features, (distances, fft_coefficients))

def featurize(cell_data, index):
    image, binary = cell_data['patches'][index], cell_data['masks'][index].astype(np.uint8)
    zernike = mahotas.features.zernike_moments(binary, max(binary.shape)/2, degree=8)
    #zernike = zernike / zernike.sum()
    haralick = mahotas.features.haralick(image.astype(np.uint16)).mean(axis=0)
    #haralick = haralick / haralick.sum()
    shape, info = shape_features(binary, 20)
    #print(f"Zernike: {zernike.shape}, Haralick: {haralick.shape}, Radial Shape: {shape.shape}")
    return(np.concatenate([zernike, haralick, shape]))

class VideoDataProcessed:
    def __init__(self, files, sequence_length=5, channel=0):
        self.data = {}
        self.all_traces = []
        self.seq_length = sequence_length
        self.channel = channel
        self.videos = {}
        for category, num in files:
            print(f"Loading in processed {num}")
            assert category == 'processed', "Can't load non processed file"
            video = get_file(category, num)
            self.videos[num] = video
        self.num_vids = len(self.data)

    def __len__(self):
        return self.num_vids

    def extract_planes(self, num, zplanes, hist_length):
        for z in zplanes:
            self.extract_slice_traces(num, z, hist_length)
    
    def extract_slice_traces(self, num, zPlane, hist_length=2):
        assert num in self.videos.keys(), f"Video {num} not found"
        
        video = self.videos[num]
        frames, shp = video.read_image(C=self.channel, S=0, Z=zPlane)
        frames = scale_img(frames.squeeze())
        print(f"vid {num} zplane {zPlane} with frames: {frames.shape}")
        masks = binarize_video(frames)
        N = len(frames)
        s = 0
        for i in range(N // self.seq_length):
            print(f"Extracting traces from {s}:{s+self.seq_length}")
            data = extract_traces_sparse(frames[s:s+self.seq_length], masks[s:s+self.seq_length], hist=hist_length)
            s += self.seq_length
            self.all_traces = self.all_traces + data
        
        if(N % self.seq_length > 0):
            data = extract_traces_sparse(frames[-1*self.seq_length:], masks[-1*self.seq_length:], hist=hist_length)
            self.all_traces = self.all_traces + data


class SparseMIPVideo:
    def __init__(self, files, sequence_length, hist_length=2):
        self.data = {}
        self.all_traces = []
        self.N = sequence_length
        for category, num in files:
            print(f"Loading in MIP {num}")
            assert category == 'mip', "Can't load non Mip file"
            video = get_file(category, num)
            frames, shp = video.read_image(C=0)
            frames = scale_img(frames.squeeze())
            print(f"frames {num}: {frames.shape}")
            masks = binarize_video(frames)

            print(f"Finished loading frames and masks for MIP {num}")

            N = len(frames)
            s = 0
        
            for i in range(N // sequence_length):
                print(f"Extracting traces from {s}:{s+sequence_length}")
                data = extract_traces_sparse(frames[s:s+sequence_length], masks[s:s+sequence_length], hist=hist_length)
                s += sequence_length
                self.all_traces = self.all_traces + data
            
            if(N % sequence_length > 0):
                data = extract_traces_sparse(frames[-1*sequence_length:], masks[-1*sequence_length:], hist=hist_length)
                self.all_traces = self.all_traces + data

    def featurize_traces(self):
        self.featurized_frames = []
        for i, trace in enumerate(self.all_traces):
            if(i % 100 == 0):
                print(i)
            trajectory_features = np.array([featurize(trace, index) for index in range(5)])
            self.featurized_frames.append(trajectory_features)