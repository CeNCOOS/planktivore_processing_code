import datetime as dt
import pytz
import pdb
def date_from_name_oct2025(filename,magnification):
    """
    Extracts the date from a planktivore filename'.
    """
    timezone=pytz.timezone('UTC')
    parts = filename.split('-')
    #try:
    magind=[idx for idx, part in enumerate(parts) if magnification in part][0]
    # need to add some code for Oct 2025 deployment with clock problem
    # start time 1761589506572844
    # 16 digits
    # seconds is 10 digits
    # offset=-5180 seconds
    # so for this timestamp -5180000000
    start_time=1761589506572844
    #pdb.set_trace()
    delta=int(parts[magind+1])-start_time
    drift=(delta/1000000)*6.4231e-5
    offset=-5180000000
    newvalue=int(parts[magind+1])+offset-(drift*1000000)
    ltime=int(newvalue)/1000000
    #ltime=int(parts[magind+1])/1000000
    imagetime=dt.datetime.fromtimestamp(ltime).astimezone(timezone)
    return imagetime
    #except:
    #    return None
