import sys
import os
import json
import multiprocessing as mp
from glob import glob
import tarfile
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time
import pdb
#sys.path.append('u:/planktivore')
import cvtools_modified as cvtools
from check_dir import check_dir
from date_from_name import date_from_name
# 
def convert_tif_to_rawcolor(thedir,outputdir,subdirmins,maglev,settings):
    # convert_tif_to_rawcolor(inputdir,outputdir,subdir, maxfiles,maglev,settings,ncount,dircount,lock1,lock2)
    # inputdir: Directory with images to convert
    # outputdir: Directory for image output
    # subdir: structure of subdirectories under outputdir
    # maxfiles: Maximum number of files to write per directory/subdir to avoid ls crash between 10 and 50k
    # maglev: magnification level (high vs. low)
    # settings: rims style settings
    filedate=date_from_name(thedir,maglev) # get the date from the file name
    if filedate is None:
        print(f"File {thedir} does not contain a valid date.")
        return
    elif filedate is not None:
        subdir_path=check_dir(outputdir,filedate,subdirmins) # check if the directory exists and create it if not
    #current_subdir=dircount.value # start with the current subdirectory count
    #file_count=ncount.value # start with the current file count
    #base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
    #os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist
        fsize=cv2.imread(thedir)
        if fsize is not None:
        #fsize=os.stat(thedir).st_size # get the file size
        #if fsize > 10:
        # determine if we need to create a new subdirectory
        #if file_count >= maxfiles:
        #    with lock2:
        #        dircount.value += 1
        #        current_subdir = dircount.value
        #        with lock1:
        #            ncount.value=0
        #            file_count=ncount.value
    # create the subdirectory
        #asubdir=subdir+f'{current_subdir}'
        #subdir_path=os.path.join(base_output_dir,asubdir) # create the subdirectory path
        # subdir_path=os.path.join(base_output_dir,f'subdir_{current_subdir}') # create the subdirectory path
        #os.makedirs(subdir_path,exist_ok=True) # create the subdirectory if it doesn't exist
        # create the subdirectory path
        # Not sure how universal this is but it works for the current data
            idslash=thedir.rfind('/')
            abpath=thedir[0:idslash+1]
            ftif=thedir[idslash+1:]
            img=cvtools.import_image(abpath,ftif,settings)
            img_c_8bit=cvtools.convert_to_8bit(img,settings)
            fout=ftif[:-4]
            output=cvtools.extract_features(img_c_8bit,img,settings,
                                            save_to_disk=True,
                                            abs_path=os.path.join(subdir_path),file_prefix=fout)
            # update the file count and lock value so multiprocessing works correctly.
        #with lock1:
        #    ncount.value += 1
        #    file_count=ncount.value
    return #file_count,current_subdir

def convert_tar_to_rawcolor(thedir,outputdir,subdirmins,maglev,settings,bayer_pattern):
        try:
            tobj=tarfile.open(thedir,'r')
        # the tar file could literally just have a directory and no files or data in it.
            tnames=tobj.getmembers()
            numfiles=np.arange(1,len(tnames))
            #current_subdir=dircount.value # start with the current subdirectory count
            #file_count=ncount.value # start with the current file count
        #base_output_dir=os.path.join(outputdir,subdir) # this is the base output directory
            #base_output_dir=outputdir
            #os.makedirs(base_output_dir,exist_ok=True) # create the base output directory if it doesn't exist      
            for afile in numfiles:
                # check if we need to create a new subdirectory
                #if ncount.value >= maxfiles:
                #    with lock2:
                #        dircount.value += 1
                #        current_subdir = dircount.value
                #        with lock1:
                #            ncount.value=0
                #            file_count=ncount.value
                #asubdir=subdir+f'{current_subdir}'
               #subdir_path=os.path.join(base_output_dir,asubdir) # create the subdirectory path
            #subdir_path=os.path.join(base_output_dir,f'subdir_{current_subdir}') # create the subdirectory path 
                #os.makedirs(subdir_path,exist_ok=True) # create the subdirectory if it doesn't exist
            # create the subdirectory path  
                tmpf=tnames[afile].name
                filetime=date_from_name(tmpf,maglev) # get the date from the file name
                if filetime is None:
                    print(f"File {tmpf} does not contain a valid date.")
                    continue
                elif filetime is not None:
                    subdir_path=check_dir(outputdir,filetime,subdirmins)
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
                #with lock1:
                #    ncount.value += 1
                #    file_count=ncount.value
        except:
            print('Failure')
            print(f"Error processing {thedir}")
            #pdb.set_trace()

        return #file_count,current_subdir
# 
if __name__ == "__main__":
# main
# get some variables needed for processing the images
    fid=open('c:/users/flbahr/setup_process_planktivore_refact.json','r')
    plset=json.load(fid)
    fid.close()
    # Now set variables based upon jason file data
    pvtr_settings=plset[0]['pvtr_setting']
    maglev=plset[0]['maglev']
    input_dir=plset[0]['inputdir']
    output_dir=plset[0]['outputdir'] # this is the root output directory
    # want to change this to the number of minutes of data files to put in a directory.
    subdirmins=plset[0]['subdirmins'] # this is the number of minutes of data files to put in a directory
   # subdir=plset[0]['subdir']
    # This will probably go away as max files is NOT how we will be working this...
    #maxfiles=plset[0]['maxfiles']
    bayer_pattern=plset[0]['bayer_pattern']
    # find all the directories within this dir
    dirs=glob(input_dir+"/**",recursive=True) # this only grabs with the * and not *.*
    dirs=sorted(dirs)
    # not sure which we want to use
    jf=open(pvtr_settings) # open the json file)
    settings=json.load(jf)
    jf.close()
    # get the atual pattern value from the code
    bayer_pattern=getattr(cv2,bayer_pattern)
    # Now we want to set up the loop for the directories to run through and the code
    # to not process files that are not tar or tifs
    lentotal=len(dirs) # get the number of directories
    mylist=[]
    # 'i' indicates we are using integers
    # Won't be using locks and counts for subdirectories.

    #ncount=mp.Value('i',0) # set the number of files to 0
    #dircount=mp.Value('i',0) # set the number of directories to 0
    #lock1=mp.Lock() # create a lock for the file count
    #lock2=mp.Lock() # create a lock for the directory count
    processes=[] # create a list of processes
    # Loop through the directories and files and set up processing
    for adir in range(0,50):
    #for adir in range(0,lentotal):
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
        #
        # Now the file should be either a tar file or a tif file I think.
        #
        if os.path.splitext(thedir)[1]=='.tar':
            # thedir is full path to the tar file
            # input dir is directory from which this was called but the base_directory
            # output is where we want to write to.
            start_time=time.time()
            #convert_tar_to_rawcolor(thedir,output_dir,subdirmins,maglev,settings,bayer_pattern)
            try:
                process=mp.Process(target=convert_tar_to_rawcolor, args=(thedir,output_dir,subdirmins,maglev,settings,bayer_pattern))
                processes.append(process)
                process.start()
            except:
                print('Failure')
                #pdb.set_trace()
            # old call kept for if we need it.
            #[ncount,dircount]=convert_tar_to_rawcolor(thedir,output_dir,subdir,maxfiles,settings,bayer_pattern,ncount,dircount)
            end_time=time.time()
            elapsed_time=end_time-start_time 
            print(f"Time taken to process {thedir}: {elapsed_time:.2f} seconds")
        if os.path.splitext(thedir)[1]=='.tif':
            #convert_tif_to_rawcolor(thedir,output_dir,subdirmins,maglev,settings)
            process=mp.Process(target=convert_tif_to_rawcolor, args=(thedir,output_dir,subdirmins,maglev,settings))
            processes.append(process)
            process.start()
    for process in processes:
        process.join()
