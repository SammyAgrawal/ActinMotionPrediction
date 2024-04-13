# index is which video choosing (6th one, etc)
from aicspylibczi import CziFile
from google.cloud import storage
from os import cpu_count
from os import mkdir
from concurrent.futures import ThreadPoolExecutor

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

        self.loaded_czifile_dict = dict()
    
    
    def download_blob(source_blob_name, destination_file_name):
        storage_client = storage.Client(project='jal2340-applied-cv-s24')
        bucket = storage_client.bucket("3d-dicty-data")
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")


    # MIP: Maximum Intensity Projection
    # multi: multiprocessing for speed increase
    def get_all(self, type = "MIP", folder = "MIPDownloaded", workers=5):
 
        mkdir(folder)

        blobs_to_load = list() 
        
        for date in self.experiment_dates:
            vid_index_list = list(range(len(self.blob_dict[date][type]))) 
            for vid_index in vid_index_list:
                    blob = self.blob_dict[date][type][vid_index]
                    download_fname = blob.name.split('/')[-1] # ignore date/fname
                    blobs_to_load.append((blob.name, folder + '/' + download_fname))
        
        # Using ThreadPoolExecutor to download files in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for blob_name, dest_name in blobs_to_load:
                executor.submit(CZIBucketInterface.download_blob(blob_name, dest_name))
