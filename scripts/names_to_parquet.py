import pandas as pd
import numpy as np
import json
import os
import tarfile
from glob import glob
import pdb


if __name__=="__main__":
    json_file="c:/users/flbahr/documents/github/planktivore_processing_code/json_files/setup_process_planktivore_lowmag_oct2024.json"
#    json_file="c:/users/flbahr/documents/github/planktivore_processing_code/json_files/setup_process_planktivore_highmag_jun2025.json"
#    json_file=input("Enter the path to the JSON file with settings: ")
    try:
        fid=open(json_file,'r')
    except Exception as e:
        print(f"Error opening JSON file: {repr(e)}")
    plset=json.load(fid)
    fid.close()
    maglev="low_mag_cam"
#    maglev="high_mag_cam"
    #maglev=plset[0]['maglev']

    input_dir=plset[0]['inputdir']
    # find all the directories within this dir
    dirs=glob(input_dir+"/**",recursive=True) # this only grabs with the * and not *.*
    dirs=sorted(dirs)
    lentotal=len(dirs) # get the number of directories
    mylist=[]
    nf=np.arange(0,lentotal)
    filenames=[]
    for adir in nf:
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
            try:
                tobj=tarfile.open(thedir,'r')
            # the tar file could literally just have a directory and no files or data in it.
                tnames=tobj.getmembers() # this is a list so we don't want to append list to list we want to append elements to list
            # get partial names
                part1=[test.name for test in tnames]
                part2=[name for name in part1 if name.endswith(".tif")]
                part3=[os.path.basename(name) for name in part2]
            
                filenames.extend(part3)
            except Exception as e:
                print(f"Failur to process {thedir}")
                print(repr(e))
            #pdb.set_trace()
                #numfiles=np.arange(1,len(tnames))
        if os.path.splitext(thedir)[1]=='.tif':
            part3=os.path.basename(thedir)
            filenames.append(part3)
    df=pd.DataFrame(filenames)
    df.to_parquet('October_2024_Ahi_lowmag.parquet',engine='pyarrow')
        
        
