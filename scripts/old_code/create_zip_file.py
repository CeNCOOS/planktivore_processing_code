import zipfile
import re
import cv2
import pandas as pd
# read the file with the image names
filetoread='/home/flbahr/fredmorpho/synchro_cluster.csv'
fid=open(filetoread,'r')
# create the index.csv file to write to.
indexfile='/home/flbahr/index.csv'
fio=open(indexfile,'w')
# write required first line of index.csv file
fio.write('object_id,path\n')
lines=fid.readlines()
# loop through the names
for iv in lines:
    # find the slash location in the path name
    slashindex=[x.start() for x in re.finditer('/',iv)]
    # grab just the image name
    imagename=iv[slashindex[-1]+1:-1]
    fio.write(imagename+','+imagename+'\n')
fio.close()
# create and open the zip file
fileout='/home/flbahr/fredmorpho/synchro_highres_202404.zip'
zipf=zipfile.ZipFile(fileout,'w')
# write the index file to the archive
# can't have path before file name as it adds that too the archive
# morphocluster can't find the file if it has a path to the file.
zipf.write('index.csv')
## now to loop through the images and load and write them out
ic=0
for im in lines:
    slashindex=[x.start() for x in re.finditer('/',im)]
    imagename=im[slashindex[-1]+1:-1]
    # read the image
    img=cv2.imread(im[:-1])
    retv,buf=cv2.imencode('.jpg',img)
    zipf.writestr(imagename,buf)
zipf.close()
