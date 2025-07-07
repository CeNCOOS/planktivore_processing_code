#
import pandas as pd
import os
import pdb
# check if a subdirectory exist and if not create it?
def check_dir(basedir,filedate,maxminutes):
    # reformat filedate to a timestring and also check if the time is within x minutes of maxminutes
    # round time to nearest maxminutes.
    ptime=pd.to_datetime(filedate)
    rtime=ptime.round(f'{maxminutes}min')
    # check if rounded time is below ptime or above.
    if rtime < ptime:
        #rtime = rtime - pd.Timedelta(minutes=int(maxminutes))
        rtime=rtime
    elif rtime >= ptime:
        rtime = rtime - pd.Timedelta(minutes=int(maxminutes))
    rstime=rtime.strftime('%Y%m%dT%H%M%S')
    # create the directory path
    dirpath=os.path.join(basedir,rstime)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath
