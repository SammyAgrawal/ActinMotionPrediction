# use the cvenv to run this file for downloading from the google storage bucket
#   1. conda activate cvenv   
#   2. go into jlosertutils and run bash setup_paths.sh
#   2. adjust arguments to match the type of the desired data and folder to which to download 
#   4. verify whether you just want a random subset of the data (num = n), all of it (num=-1)
#   5. adjust the num workers (threads) you intend to use for your loading

import jlosertutils.czi_bucket_interface as bi

interface = bi.CZIBucketInterface()
interface.get_files(type="MIP",folder="/mnt/datadisk/FactinMIP",workers=5,random_order=False,num=-1)
