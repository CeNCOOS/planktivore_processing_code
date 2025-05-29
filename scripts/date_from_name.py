import datetime as dt
import pytz
def date_from_name(filename,mangnification):
    """
    Extracts the date from a planktivore filename'.
    """
    timezone=pytz.timezone('UTC')
    parts = filename.split('-')
    try:
        magind=parts.index(mangnification)
        ltime=parts[magind+1]/1000000
        imagetime=dt.datetime.fromtimestamp(x).astimezone(timezone)
        return imagetime
    except:
        # file does not contain a time.
        return None