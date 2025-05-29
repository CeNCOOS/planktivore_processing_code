import datetime as dt
import pytz
import pdb
def date_from_name(filename,magnification):
    """
    Extracts the date from a planktivore filename'.
    """
    timezone=pytz.timezone('UTC')
    parts = filename.split('-')
    try:
        #pdb.set_trace()
        magind=[idx for idx, part in enumerate(parts) if magnification in part][0]

        #magind=parts.index(magnification)
        ltime=int(parts[magind+1])/1000000
        imagetime=dt.datetime.fromtimestamp(ltime).astimezone(timezone)
        return imagetime
    except:
        # file does not contain a time.
        return None