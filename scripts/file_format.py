import sys
import os
import json
import multiprocessing as mp
import io 
from glob import glob
import tarfile
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import cv2
from PIL import Image
from pathlib import Path
import time
import pdb
sys.path.append('u:/planktivore')
import cvtools
# 
#
#
def convert_tif_to_rawcolor(thedir,outputdir,subdir,maxfiles,settings,ncount,dircount):
    # convert_tif_to_rawcolor(inputdir,outputdir,subdir, maxfiles,maglev,settings,ncount,dircount)
    # inputdir: Directory with images to convert
    # outputdir: Directory for image output
    # subdir: structure of subdirectories under outputdir
    # maxfiles: Maximum number of files to write per directory/subdir to avoid ls crash between 10 and 50k
    # maglev: magnification level (high vs. low)
    # settings: rims style settings
    # ncount: number of files already processed
    # dircount: number of directories already processed
    current_subdir=dircount # start with the current subdirectory count
    file_count=ncount
    base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
    os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist
    fsize=os.stat(thedir).st_size # get the file size
    if fsize > 0:
        if file_count >= maxfiles:
            current_subdir += 1 # increment the subdirectory count
            file_count = 0 # reset the file count
    # create the subdirectory
            subdir_path=os.path.join(base_output_dir,f'subdir_{current_subdir}') # create the subdirectory path
            os.makedirs(subdir_path,exist_ok=True) # create the subdirectory if it doesn't exist
            # create the subdirectory path
            idslash=thedir.rfind('/')
            abpath=thedir[0:idslash+1]
            ftif=thedir[idslash+1:]
            img=cvtools.import_img(abpath,ftif,settings)
            img_c_8bit=cvtools.convert_to_8bit(img,settings)
            fout=ftif[:-4]
            output=cvtools.extract_features(img_c_8bit,img,settings,
                                                save_to_disk=True,
                                                abs_path=os.path.join(subdir_path),file_prefix=fout)
            file_count += 1 # increment the file count
    return file_count,current_subdir

def convert_tar_to_rawcolor(thedir,outputdir,subdir,maxfiles,settings,bayer_pattern,ncount,dircount):
        tobj=tarfile.open(thedir,'r')
        # the tar file could literally just have a directory and no files or data in it.
        tnames=tobj.getmembers()
        numfiles=np.arange(1,len(tnames))
        current_subdir=dircount
        file_count=ncount
        base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
        os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist      
        for afile in numfiles:
            if file_count >= maxfiles:
                current_subdir += 1 # increment the subdirectory count
                file_count = 0 # reset the file count   
            subdir_path=os.path.join(base_output_dir,f'subdir_{current_subdir}') # create the subdirectory path 
            os.makedirs(subdir_path,exist_ok=True) # create the subdirectory if it doesn't exist
        # create the subdirectory path  
            tmpf=tnames[afile].name
            # remove the dirctory at start of name?
            sslash=tmpf.rfind('/')
            nfout=tmpf[sslash+1:]
            idslash=nfout.rfind('\\')
            fout=nfout[idslash+1:-4]
            subdir_path=subdir_path.replace("\\","/")
            fileobj=tobj.extractfile(tnames[afile])
            tiff_array=np.asarray(bytearray(fileobj.read()),dtype=np.uint8)
            img=cv2.imdecode(tiff_array,cv2.IMREAD_UNCHANGED)
            imgc=cv2.cvtColor(img,bayer_pattern)
            img_c_8bit=cvtools.convert_to_8bit(imgc,settings)
            output=cvtools.extract_features(img_c_8bit,imgc,settings,save_to_disk=True,
                                            abs_path=subdir_path,file_prefix=fout)
            file_count += 1 # increment the file count

        return file_count,current_subdir
# 
if __name__ == "__main__":
# main
# get some variables needed for processing the images
fid=open('c:/users/flbahr/setup_process_planktivore.json','r')
plset=json.load(fid)
fid.close()
# Now set variables based upon jason file data
pvtr_settings=plset[0]['pvtr_setting']
maglev=plset[0]['maglev']
input_dir=plset[0]['inputdir']
output_dir=plset[0]['outputdir']
subdir=plset[0]['subdir']
maxfiles=plset[0]['maxfiles']
bayer_pattern=plset[0]['bayer_pattern']
# find all the directories within this dir
#diru=glob(input_dir+"/**") # this only grabs with the * and not *.*
dirs=glob(input_dir+"/**",recursive=True) # this only grabs with the * and not *.*
#diru=sorted(diru)
dirs=sorted(dirs)
# not sure which we want to use
jf=open(pvtr_settings) # open the json file)
settings=json.load(jf)
jf.close()
# get the atual pattern value from the code
bayer_pattern=getattr(cv2,bayer_pattern)
#bayer_pattern=cv2.COLOR_BAYER_RG2RGB # set the bayer pattern to use for conversion
# Now we want to set up the loop for the directories to run through and the code
# to not process files that are not tar or tifs
#lenupper=len(diru) # get the number of directories
lentotal=len(dirs) # get the number of directories
mylist=[]
ncount=0
dircount=0
# put in test range for testing 
for adir in range(0,29):
    thedir=dirs[adir]
    if os.path.splitext(thedir)[1]=='.log':
        continue
    if os.path.splitext(thedir)[1]=='.txt':
        continue   
    if os.path.splitext(thedir)[1]=='.bin':
        continue
    if len(os.path.splitext(thedir)[1]) == 0:
        continue   
    # check if we have the correct magnification level
    if thedir.find(maglev) < 0:
        continue
    # for dos machine check the direction of slashes and fix
    thedir=thedir.replace("\\","/")
    #if len(os.path.splitext(thedir)[1]) > 0:
    #    mylist.append(thedir) # add the directory to the list
    # check if the directory is a tar file or a tif file
    #
    # Now the file should be either a tar file or a tif file I think.
    #
    #pdb.set_trace()
    if os.path.splitext(thedir)[1]=='.tar':
        # thedir is full path to the tar file
        # input dir is directory from which this was called but the base_directory
        # output is where we want to write to.
        # problem is we have the full path and don't need to re glob the files.
        #
    
        #
        start_time=time.time()
        [ncount,dircount]=convert_tar_to_rawcolor(thedir,output_dir,subdir,maxfiles,settings,bayer_pattern,ncount,dircount)
        end_time=time.time()
        elapsed_time=end_time-start_time
    
        print(f"Time taken to process {thedir}: {elapsed_time:.2f} seconds")
        #pdb.set_trace()

        #print(thedir)
        # pass back how many files were written.
        #convert_tar_to_rawcolor(inputdir,output_dir,subdir,maxfiles,maglev,settings,bayer_pattern)
        #pass
    if os.path.splitext(thedir)[1]=='.tif':
        [ncount,dircount]=convert_tif_to_rawcolor(thedir,output_dir,subdir,maxfiles,settings,ncount,dircount)
        # convert_tif_to_rawcolor(inputdir,output_dir,subdir,maxfiles,maglev,settings)
        #pass

#pdb.set_trace()
#dirs=glob(root_data_path+"/*/") # this only grabs with the * and not *.*
# set the directory magnification to load
# this could be also input
#maglev='high_mag_cam_rois/'
#maglev='low_mag_cam_rois/'
# the following was a test case need to use above to create a functional generalization
#files=glob(dirs[12]+maglev+'**',recursive=True) # recursively go through the directory and get files
# output paths...
#full_color_output='u:/planktivore/output_test/' # can we bypass this and just take image as byte stream
# and push to rormatted file?
#full_color_reform='u:/planktivore/reformat_data/'
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Pad and rescale ROIs to square format.")
#    parser.add_argument("--input_dir", type=Path, required=True, help="Path to the input directory containing ROI images.")
#    parser.add_argument("--output_dir", type=Path, required=True, help="Path to the output directory where processed images will be saved.")
#    args = parser.parse_args()
#
#    pad_and_rescale(args.input_dir, args.output_dir)

#numfiles=np.arange(0,len(files))
#for afile in numfiles:
#    filetype1=files[afile].endswith('.tif')
#    filetype2=files[afile].endswith('.tar')
