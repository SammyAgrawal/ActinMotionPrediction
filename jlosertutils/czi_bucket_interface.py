# index is which video choosing (6th one, etc)
from aicspylibczi import CziFile
from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor
import random

# This class is useful for downloading all the videos to create a dataset
class CZIBucketInterface:
    def __init__(self):      
        self.storage_client = storage.Client(project='jal2340-applied-cv-s24')
        self.bucket = self.storage_client.bucket("3d-dicty-data")

        self.experiment_dates = ['2023-01-30', '2023-10-25'] 
        
        self.blob_dict = {
            self.experiment_dates[0] : {"MIP" : [], "raw": [], "processed":[]},
            self.experiment_dates[1] : {"MIP" : [], "raw": [], "smiley":[]},
        } 
        
        for blob in self.bucket.list_blobs():
            if(self.experiment_dates[0] in blob.name):
                if('MIP.czi' in blob.name):
                    self.blob_dict[self.experiment_dates[0]]["MIP"].append(blob)
                elif("processed.czi" in blob.name):
                    self.blob_dict[self.experiment_dates[0]]["processed"].append(blob)
                else:
                    self.blob_dict[self.experiment_dates[0]]["raw"].append(blob)
            elif(self.experiment_dates[1] in blob.name):
                if('MIP.czi' in blob.name):
                    self.blob_dict[self.experiment_dates[1]]["MIP"].append(blob)
                elif("smiley.czi" in blob.name):
                    self.blob_dict[self.experiment_dates[1]]["smiley"].append(blob)
                else:
                    self.blob_dict[self.experiment_dates[1]]["raw"].append(blob)

        self.czi_list = list()
    
    
    def download_blob(source_blob_name, destination_file_name):
        storage_client = storage.Client(project='jal2340-applied-cv-s24')
        bucket = storage_client.bucket("3d-dicty-data")
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")


    # MIP: Maximum Intensity Projection
    # multi: multiprocessing for speed increase
    def get_files(self, type = "MIP", folder = "MIPDownloaded", workers = 5, random_order = True, num = -1):
        os.mkdir(folder)

        blobs_to_load = list() 
        
        dates = None
        if (type == "smiley"):
            dates = ['2023-10-25']
        elif (type == "processed"):
            dates = ['2023-01-30']
        else :
            dates = self.experiment_dates

        files_downloaded = 0
        for date in dates :
            vid_index_list = list(range(len(self.blob_dict[date][type])))
            if (random_order):
                random.shuffle(vid_index_list) 
            for vid_index in vid_index_list:
                    if (num == files_downloaded):
                        break
                    blob = self.blob_dict[date][type][vid_index]
                    download_fname = blob.name.split('/')[-1] # ignore date/fname
                    blobs_to_load.append((blob.name, folder + '/' + download_fname))
                    if (num > -1):
                        files_downloaded +=1
            

        # Using ThreadPoolExecutor to download files in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for blob_name, dest_name in blobs_to_load:
                executor.submit(CZIBucketInterface.download_blob(blob_name, dest_name))

    def get_czi_list(self,folder):
        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path):
                self.czi_list.append(CziFile(full_path))

    def print_czi_list_shape(self):
        for czifile in self.czi_list:
            print(czifile.get_dims_shape())
            