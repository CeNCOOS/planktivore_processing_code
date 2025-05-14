# Code for processing planktivore images from the raw tif format to the
# complete suite of images these are what are fed into RIMS I believe.
import time
import cvtools
import tarfile
import cProfile
import pstats
import multiprocessing
import json
import io
import os
import cv2
from PIL import Image
import numpy as np
from glob import glob
import pdb
# pixel pattern on the sensor.  The "tif" files are the raw pixels from the sensor.
# This pattern needs to be applied to get the actual colors for each pixel.
# This code is set to only process the high_mag_rois data
# The low_mag_rois have about 10 times as many images.
bayer_pattern=cv2.COLOR_BAYER_RG2RGB
# The JSON file contains how the images are processed.
jf=open('ptvr_proc_settings.json')
settings=json.load(jf)
jf.close()
# get the base directories
# local directories for re-processing the planktivore data 
# the whole set of outputs are kept (I think RIMS has this also but the *.jpg are
# what you can access).  These files contain in the zip file the unmasked *.jpg
# which may be a better image to use.  
basepath='/sas-array/opt/planktivore-share-copy/2024-04-15-LRAH-12/'
baseoutput='/sas-array/opt/planktivore/fprocessed/'
dirs=glob(basepath+"/*/")
ic=1
iout=1
# max number of images to write to a directory.
# linux has an issue with "inodes" when there are too many files.
# We don't want to crash the system.  
fmax=50000
#
id=len(dirs)
# loop through the directories and process the images.
for i in np.arange(0,id):
    startdir=dirs[i]
    outputdir=[]
    if len(startdir) > 70:
    # now add high or low res path
        mag='high_mag_cam_rois/'
        try:
            tars=glob(startdir+mag+'*.tar')
            for entry in os.scandir(startdir+mag):
                if entry.is_dir():
                    outputdir.append(entry.path)
            print('loop id='+str(i)+'   '+str(len(tars)))
            if len(outputdir) > 0:
                print(outputdir)
        except:
            pass
    else:
        pass
    if len(outputdir) > 0:
        # get number of files in dir
        thetifs=glob(outputdir[0]+'/*.tif')
        print('Uncompressed tiffs = '+str(len(thetifs)))
        # process the tif files
        nti=np.arange(0,len(thetifs))
        for ztif in nti:
            # first read the tiff image.  This is for the tarred images.
            fsize=os.stat(thetifs[ztif]).st_size
            if fsize > 0:
                idslash=thetifs[ztif].rfind('/')
                abpath=thetifs[ztif][0:idslash+1]
                ftif=thetifs[ztif][idslash+1:]
                # import the image from the file
                img=cvtools.import_image(abpath,ftif,settings)
                # convert to a 8 bit colore image
                img_c_8bit=cvtools.convert_to_8bit(img,settings)
                # extract the output name for writing out
                fout=ftif[13:-4]
                # create the full output with mask, roi, unmasked image etc.
                output=cvtools.extract_features(img_c_8bit,img,settings,save_to_disk=True,abs_path=baseoutput+'group'+str(iout)+'/',file_prefix=fout)
                ic=ic+1
                fmod=ic % fmax
                if fmod==0:
                    iout=iout+1
                    os.makedirs(baseoutput+'group'+str(iout))
    if len(tars) > 0:
        nta=np.arange(0,len(tars))
        # loop through the tar files and extract the tif and process similar to above
        # again this is creating a local copy that can be used to create the file for
        # morphocluster.  
        for nts in nta:
            try:
                tobj=tarfile.open(tars[nts],'r')
                tnames=tobj.getmembers()
                nnam=np.arange(1,len(tnames))
                for aname in nnam:
                    tmpf=tnames[aname].name
                    slashid=tmpf.rfind('/')
                    namefile=tmpf[slashid+1:-4]
                    fileobj=tobj.extractfile(tnames[aname])
                    tiff_array=np.asarray(bytearray(fileobj.read()),dtype=np.uint8)
                    img=cv2.imdecode(tiff_array,cv2.IMREAD_UNCHANGED)
                    imgc=cv2.cvtColor(img,bayer_pattern)
                    img_c_8bit=cvtools.convert_to_8bit(imgc,settings)
                    output=cvtools.extract_features(img_c_8bit,imgc,settings,save_to_disk=True,abs_path=baseoutput+'group'+str(iout)+'/',file_prefix=namefile)
                    ic=ic+1
                    fmod=ic % fmax
                    if fmod==0:
                        iout=iout+1
                        os.makedirs(baseoutput+'group'+str(iout))
            except:
                pass
