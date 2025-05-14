import sys
import os
import json
import multiprocessing
import io 
from glob import glob
import tarfile
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import cv2
from PIL import Image
import argparse
from pathlib import Path
import pdb
sys.path.append('u:/planktivore')
import cvtools
from adjust_roi import pad_and_rescale
# 
#
#
#def convert_tif_to_rawcolor(inputdir,outputdir,subdir,maxfiles,maglev,settings):
def convert_tif_to_rawcolor(thedir,outputdir,subdir,maxfiles,settings,ncount,dircount):
    # convert_tif_to_rawcolor(inputdir,outputdir,subdir, maxfiles,maglev,image_min_size)
    # inputdir: Directory with images to convert
    # outputdir: Directory for image output
    # subdir: structure of subdirectories under outputdir
    # maxfiles: Maximum number of files to write per directory/subdir to avoid ls crash between 10 and 50k
    # maglev: magnification level (high vs. low)
    # image_min_size: minimum size of an image to process.  Too small and it isn't of use? Actually we don't want
    # to do this.
    # Do we need to count these small images?
    # Need max files per directory
    # Need to keep track of subdirectories with the count
    # Need to limit how many files per directory
    # need to also deal with images that are "too small"
    # how to keep track of subdir # and file numbers?
    #
    #thefiles=glob(inputdir+maglev+'**',recursive=True) # find files in the directory
    # In this case we will only have 1 tif file and not a whole list.
    current_subdir=dircount # start with the current subdirectory count
    #current_subdir=0 
    file_count=ncount
    #file_count=0
    base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
    os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist
    #numfiles=np.arange(0,len(thefiles)) # total number of files
    # Do we need to check numfiles with maxfiles?
    #for afile in numfiles:
    fsize=os.stat(thedir).st_size # get the file size
        #fsize=os.stat(thefiles[afile]).st_size
    if fsize > 0:
        if file_count >= maxfiles:
            current_subdir += 1 # increment the subdirectory count
            file_count = 0 # reset the file count
    # create the subdirectory
            subdir_path=os.path.join(base_output_dir,f'subdir_{current_subdir}') # create the subdirectory path
            os.makedirs(subdir_path,exist_ok=True) # create the subdirectory if it doesn't exist
            # create the subdirectory path
            idslash=thedir.rfind('/')
                #idslash=thefiles[afile].rfind('\\')
            abpath=thedir[0:idslash+1]
                #abpath=thefiles[afile][0:idslash+1]
                #abpath=thefiles[afile][0:idslash+1]
            ftif=thedir[idslash+1:]
                #ftif=thefiles[afile][idslash+1:]
            img=cvtools.import_img(abpath,ftif,settings)
            img_c_8bit=cvtools.convert_to_8bit(img,settings)
            fout=ftif[:-4]
            output=cvtools.extract_features(img_c_8bit,img,settings,
                                                save_to_disk=True,
                                                abs_path=os.path.join(subdir_path),file_prefix=fout)
            file_count += 1 # increment the file count
    return file_count,current_subdir
    #num_workers=max(1,os.cpu_count()-1)

#def convert_tar_to_rawcolor(inputdir,outputdir,subdir,maxfiles,maglev,settings,bayer_pattern):
def convert_tar_to_rawcolor(thedir,outputdir,subdir,maxfiles,settings,bayer_pattern,ncount,dircount):
    ## need to add subdir path 
    ## need to add max files per directory
    #thefiles=glob(inputdir+maglev+'/'+'**',recursive=True)

    ## the number of tar files?
    #num_files=np.arange(0,len(thefiles))
    
    #for atfile in num_files:
    #    #Need to check if the file is a tar file
    #    filetype1=thefiles[atfile].endswith('.tar')
    #    if not filetype1:
    #        print(f"File {thefiles[atfile]} is not a tar file. Skipping...")
    #        continue   

        #tobj=tarfile.open(thefiles[atfile],'r')
        tobj=tarfile.open(thedir,'r')
        # the tar file could literally just have a directory and no files or data in it.
        tnames=tobj.getmembers()
        numfiles=np.arange(1,len(tnames))
        current_subdir=dircount
        #current_subdir=0
        file_count=ncount
        #file_count=0
        base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
        os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist      
        #pdb.set_trace()
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
            #pdb.set_trace()
            idslash=nfout.rfind('\\')
            #idslash=tmpf.rfind('\\')
        #namefile=tmpf[slashid+1:-4]
            fout=nfout[idslash+1:-4]
            #print(fout)
            #fout=tmpf[idslash+1:-4]
            subdir_path=subdir_path.replace("\\","/")
            print('subdir_path='+subdir_path)
            fileobj=tobj.extractfile(tnames[afile])
            tiff_array=np.asarray(bytearray(fileobj.read()),dtype=np.uint8)
            img=cv2.imdecode(tiff_array,cv2.IMREAD_UNCHANGED)
            #pdb.set_trace()
            imgc=cv2.cvtColor(img,bayer_pattern)
            img_c_8bit=cvtools.convert_to_8bit(imgc,settings)
            #pdb.set_trace()
            #print(fout)
            #output=cvtools.extract_features(img_c_8bit,imgc,settings,save_to_disk=True,
            #                                abs_path=os.path.join(subdir_path,fout))
            output=cvtools.extract_features(img_c_8bit,imgc,settings,save_to_disk=True,
                                            abs_path=subdir_path,file_prefix=fout)
        return file_count,current_subdir
# 
# Is this where we put the main code?

# main
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

#root_data_path='Y:/2025-04-14-LRAH-27/'  #this could be part of the arg input like what Danelle uses
# find all the directories within this dir
#dirs=glob(input_dir+"2025-04-16-02-11-23.030364673/*") # this only grabs with the * and not *.*
diru=glob(input_dir+"/**") # this only grabs with the * and not *.*
dirs=glob(input_dir+"/**",recursive=True) # this only grabs with the * and not *.*
diru=sorted(diru)
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
lenupper=len(diru) # get the number of directories
lentotal=len(dirs) # get the number of directories
mylist=[]
ncount=0
dircount=0
#pdb.set_trace()
# put in test range for testing 
for adir in range(0,29):
#for adir in range(16,lentotal):
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
        [ncount,dircount]=convert_tar_to_rawcolor(thedir,output_dir,subdir,maxfiles,settings,bayer_pattern,ncount,dircount)
        print(thedir)
        # pass back how many files were written.
        #convert_tar_to_rawcolor(inputdir,output_dir,subdir,maxfiles,maglev,settings,bayer_pattern)
        #pass
    if os.path.splitext(thedir)[1]=='.tif':
        [ncount,dircount]=convert_tif_to_rawcolor(thedir,output_dir,subdir,maxfiles,settings,ncount,dircount)
        # convert_tif_to_rawcolor(inputdir,output_dir,subdir,maxfiles,maglev,settings)
        #pass

pdb.set_trace()
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