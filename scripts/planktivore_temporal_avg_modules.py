import pandas as pd
import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cm
from pathlib import Path
import seaborn as sns
import tqdm
#import rasterio
#from rasterio.plot import reshape_as_raster
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
from scipy.signal import find_peaks, savgol_filter
import datetime as dt
from scipy.stats import linregress
#
# Modules needed for temporal average of data from planktivore
#
def read_image_names(fpath,fname):
    '''
    Read the filenames from a file and parse the time from the filename
    input:
        fpath: path to the file with the list of names
        fname: parquet file containing the names
    output:
        df: dataframe of the file times and names
    '''
    df=pd.read_parquet(fpath+fname)
    df.rename(columns={0: "fileName"}, inplace=True)
    df['dateStr'] = df["fileName"].str.split("-",expand=True)[1]
    numeric_microseconds=pd.to_numeric(df['dateStr'])
    df['dateTime'] = pd.to_datetime(numeric_microseconds, unit='us')
    df['dateTime'] = df['dateTime'].dt.tz_localize('UTC').dt.tz_localize(None)
    df.index = df['dateTime']
    df.reset_index(drop=True, inplace=True)
    return df

#
# The first set of modules are to clean up the data for Synchro style processing
# For Synchro style processing we want profiles and want to ignore/remove
# regions where the vehicle stays at a fixed depth.  That data is definitely
# valid but for these purposes we want to remove that data.
#

# Gemini code
def find_flat_line_regions(
    df: pd.DataFrame, 
    column_name: str, 
    threshold: float, 
    window_size: int, 
    max_std: float
) -> pd.Index:
    """
    Identifies indices where the data is flat and below a specific threshold.

    Args:
        df: The input pandas DataFrame.
        column_name: The name of the column ('depth') to analyze.
        threshold: The maximum value the data can reach (e.g., 100).
        window_size: The rolling window size for calculating flatness (e.g., 10).
        max_std: The maximum allowable standard deviation for a 'flat' segment (e.g., 0.5).

    Returns:
        A pandas Index object containing the indices of the flat regions.
    """
    
    # 1. Check the 'below threshold' condition (a simple boolean Series)
    # This filters out all data points that are above the general threshold.
    is_below_threshold = df[column_name] < threshold

    # 2. Check the 'flatness' condition using Rolling Standard Deviation (STD)
    # Small standard deviation implies low variability (i.e., flat line).
    # We use .rolling(window_size, center=True) to look both backward and forward.
    rolling_std = df[column_name].rolling(
        window=window_size, 
        min_periods=1, # Ensure we get results near the start/end
        center=True
    ).std()
    
    is_flat = rolling_std > max_std

    # 3. Combine both conditions
    # We want indices where BOTH conditions are True.
    flat_regions = df.index[is_below_threshold & is_flat]
    
    return flat_regions
#
# 
#
def label_casts_by_turning_points(
    df,
    depth_col="depth",
    smooth=True,
    win_len=21,         # odd integer; tune as neeed
    polyorder=2,
    peak_prominence=0.25,  # meters; tune to ignore tiny wiggles in depth
    peak_distance=10       # samples; min spacing between turning points
):
    """
    Label AUV casts using turning points (depth minima/maxima).
    Returns a copy of df with:
      - 'phase': 'down'|'up'|'turn'
      - 'cast_id': Int64 (new cast at the start of each 'down' segment)
    Assumes DateTimeIndex and depth positive downward.
    """
    out = df.sort_index().copy()
    depth = out[depth_col].to_numpy()
    n = len(out)
    if n == 0:
        return out.assign(phase=np.nan, cast_id=pd.Series(dtype="Int64"))

    # --- optional smoothing: Savgol_filter - Fits polynomial over a window ---
    if smooth and n >= 7:
        wl = win_len if n >= win_len else max(3, (n//2)*2 + 1)
        depth_s = savgol_filter(depth, window_length=wl, polyorder=polyorder, mode="interp")
    else:
        depth_s = depth.copy()

    # --- turning points ---
    deep_idx, _    = find_peaks( depth_s, prominence=peak_prominence, distance=peak_distance)   # local maxima (deep)
    shallow_idx, _ = find_peaks(-depth_s, prominence=peak_prominence, distance=peak_distance)   # local minima (shallow)
    turn_idx = np.unique(np.sort(np.r_[deep_idx, shallow_idx]))

    # Ensure we include start/end as implicit boundaries if needed
    if len(turn_idx) == 0 or turn_idx[0] != 0:
        turn_idx = np.r_[0, turn_idx]
    if turn_idx[-1] != n-1:
        turn_idx = np.r_[turn_idx, n-1]

    phase = np.array(["turn"]*n, dtype=object)
    cast_id = np.full(n, np.nan)
    current = 0

    # --- label each segment between turning points ---
    for a, b in zip(turn_idx[:-1], turn_idx[1:]):
        if b <= a: 
            continue
        seg = slice(a, b+1)  # inclusive segment

        # slope sign from endpoints (robust for monotonic segments)
        going_deeper = (depth_s[b] > depth_s[a])
        if going_deeper:
            phase[seg] = "down"
            # start a new cast at the *beginning* of each down segment
            current += 1
            cast_id[seg] = current
        else:
            phase[seg] = "up"
            if current > 0:
                cast_id[seg] = current

    out["phase"] = phase
    out["cast_id"] = pd.Series(cast_id, index=out.index).astype("Int64")
    out["turning_point"] = False
    out.loc[out.index[turn_idx], "turning_point"] = True
    return out
#
#
#
def rolling_r_squared(series_x: pd.Series, series_y: pd.Series, window: int) -> pd.Series:
    """
    Calculates the R-squared (coefficient of determination) for a linear fit
    within a rolling window, adapted for speed using NumPy and SciPy.
    """
    # Create an array of R-squared values, initialized to NaN
    r_squared_values = np.full(len(series_y), np.nan)
    
    # Iterate through the series, calculating linregress for each window
    # Note: Rolling functions can be complex to optimize purely with built-ins
    # We iterate, but use fast NumPy/SciPy operations within the loop.
    for i in range(len(series_y) - window + 1):
        x_window = series_x.iloc[i : i + window]
        y_window = series_y.iloc[i : i + window]
        
        # Check for minimum variability required for linregress to work
        # If x is constant (dx=0) or y is constant (dy=0), linregress may fail/be trivial
        if x_window.std() == 0:
             r_squared_values[i + window - 1] = 0.0 # Assign R^2 = 0 for constant x
             continue

        # Perform the linear regression
        # slope, intercept, r_value, p_value, std_err = linregress(...)
        result = linregress(x_window, y_window)
        
        # R-squared is r_value**2
        r_squared_values[i + window - 1] = result.rvalue**2

    # Convert the resulting array back to a Series
    return pd.Series(r_squared_values, index=series_y.index).shift(-(window-1))


def detect_interpolated_segments_rolling(
    df: pd.DataFrame, 
    value_col: str, 
    index_col: str, 
    window_size: int, 
    min_r_squared: float = 0.999
) -> pd.Series:
    """
    Detects segments where data is highly linear (likely interpolated) 
    over a defined rolling window size.

    Args:
        df: The input pandas DataFrame.
        value_col: The column containing the data values (Y-axis).
        index_col: The column containing the index/depth/time (X-axis).
        window_size: The number of consecutive points to check for linearity (e.g., 5).
        min_r_squared: The R^2 threshold to define a segment as 'linear'. 
                       (e.g., 0.999 for very strict linearity).

    Returns:
        A boolean Series where True indicates the point belongs to a highly linear segment.
    """
    if window_size < 3:
        raise ValueError("Window size must be 3 or greater for meaningful linear fit.")
        
    # 1. Calculate the Rolling R-squared
    r_squared = rolling_r_squared(
        df[index_col], 
        df[value_col], 
        window=window_size
    )

    # 2. Identify segments with a near-perfect linear fit
    is_highly_linear = r_squared >= min_r_squared

    # 3. Spread the 'True' flag across the entire window
    # The R^2 value is calculated for the *end* of the window, but we want 
    # all points within that window to be flagged.
    
    # The easiest way to flag all points in a highly linear region is to 
    # check for *any* True within the surrounding window.
    
    # We use a rolling *max* on the boolean series, which will set a value to True
    # if ANY value in the preceding window was True.
    # We apply this twice: once forward (max-window to current) and once backward (current to max+window)
    # to cover the full range of the linear segment.

    # Flag points belonging to a linear segment (forward check)
    is_linear_forward = is_highly_linear.rolling(window=window_size, min_periods=1).max().astype(bool)
    
    # Flag points belonging to a linear segment (backward check)
    is_linear_backward = is_highly_linear.rolling(window=window_size, min_periods=1).max().shift(-(window_size - 1), fill_value=False).astype(bool)
    
    # Combine the forward and backward checks
    is_interpolated = is_linear_forward | is_linear_backward
    
    return is_interpolated
#
# Read the camera log files
#
#
import sys
sys.path.append('c:/planktivore_code/')
import glob
from decode_pvtr_camlog import DualMagLog
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
#
# file name is something like YYYY-MM-DD-HH-MM-SS.SSSSS.log
#
# Need to make this a module
#
# mypath='y:/2025-04-14-LRAH-27/'
# extension='*.log'
def read_planktivore_camlogs(mypath,extension):

    filelist=glob.glob(os.path.join(mypath,'**',extension),recursive=True)
    filelist=sorted(filelist)
    nf=np.arange(0,len(filelist))
    for mf in nf:
        instance=DualMagLog(filelist[mf])
        data=instance.parse_lines()
        tmpdf=instance.export('mytest.csv')
        if mf==0:
            camdf=tmpdf
        else:
            camdf=pd.concat([camdf,tmpdf],ignore_index=True)
    # return a dataframe with the camera log information
    return camdf

def bin_casts_avg_with_time_updown(
    df: pd.DataFrame,
    depth_col: str = "depth",
    cast_col: str = "cast_id",
    time_col: Optional[str] = None,    # if None, use datetime index
    bin_size: float = 1.0,
    phase_labels: Tuple[str, str] = ("down", "up"),
    smooth_window: int = 15,            # samples (rolling median) to stabilize direction
    closed: str = "right",             # Interval closure for pd.cut
    per_phase_edges: bool = False,     # if True, compute bin edges separately for up/down
) -> pd.DataFrame:
    """
    For each cast, split trajectory into down/up phases, bin by depth, and compute:
      - numeric means per bin,
      - start_time, end_time (first/last timestamps observed in the bin),
      - elapsed_seconds (sum of dt while consecutive samples stay in the same bin),
      - rep_timestamp (timestamp of sample closest to bin center).

    Returns a tidy DataFrame indexed by [cast_id, phase, depth_bin_center].
    """
    def custom_sum_with_min_counter(series):
        return series.sum(min_count=1)

    # ---- Prep time column ----
    work = df.copy()
    if time_col is None:
        if not np.issubdtype(work.index.dtype, np.datetime64):
            raise ValueError("time_col is None but index is not datetime-like.")
        work["__time__"] = work.index
        time_col = "__time__"
    else:
        if not np.issubdtype(work[time_col].dtype, np.datetime64):
            work[time_col] = pd.to_datetime(work[time_col])

    # ---- Basic checks ----
    for c in (cast_col, depth_col, time_col):
        if c not in work.columns:
            raise ValueError(f"Missing required column: {c}")

    # The code below removes NaNs but we don't want them to become zeros.  We want NaNs
    ## ---- Clean & sort ----
    #work = (
    #    work.dropna(subset=[cast_col, depth_col, time_col])
    #        .sort_values([cast_col, time_col])
    #)

    out_frames = []

    # ---- Process each cast ----
    for cid, g_cast in work.groupby(cast_col, sort=True):
        # Ensure numeric columns are float64 for proper NaN handling
        numerical_cols=['fix_latitude', 'fix_longitude', 'salinity', 'temperature',
       'mass_concentration_of_oxygen_in_sea_water',
       'bin_mean_sea_water_salinity', 'bin_median_sea_water_salinity',
       'bin_mean_sea_water_temperature', 'bin_median_sea_water_temperature',
       'PAR', 'chlorophyll', 'chl', 'bbp470', 'bbp650',
       'volumescatcoeff117deg470nm', 'volumescatcoeff117deg650nm',
       'particulatebackscatteringcoeff470nm',
       'particulatebackscatteringcoeff650nm',
       'fix_residual_percent_distance_traveled_DeadReckonUsingMultipleVelocitySources',
       'latitude', 'longitude', 'depth', 'cast_id',
       'cast_id_unique', 'distance_km', 'rois_count', 'rois_norm', 'Akashiwo',
       'Amphidinium_Oxyphysis', 'Ceratium', 'Chaetoceros', 'Ciliate',
       'Cylindrotheca', 'Detonula_Cerataulina_Lauderia', 'Detritus',
       'Dictyocha', 'Dinoflagellate', 'Eucampia', 'Guinardia_Dactyliosolen',
       'Gyrodinium', 'Medium_pennate', 'Mesodinium', 'Mixed_diatom_chain',
       'Nano_plankton', 'Polykrikos', 'Prorocentrum', 'Pseudo-nitzschia',
       'Strombidium', 'Thalassionema', 'Thalassiosira', 'Tiarina', 'Truncated',
       'unknown_flagellate']
        g_cast[numerical_cols]=g_cast[numerical_cols].astype('float64')
        # end of new code

        g_cast = g_cast.sort_values(time_col).copy()

        # Direction: positive depth change => down; negative => up
        ddepth = g_cast[depth_col].diff()
        if smooth_window and smooth_window > 1:
            ddepth = ddepth.rolling(smooth_window, center=True, min_periods=1).median()

        sign = np.sign(ddepth).replace({0: np.nan})
        sign = sign.ffill().bfill()
        phase_map = {1.0: phase_labels[0], -1.0: phase_labels[1]}
        g_cast["phase"] = sign.map(phase_map)

        if g_cast["phase"].isna().all():
        #    # Degenerate (flat) cast — skip
            continue

        # Bin edges (shared across phases for this cast unless per_phase_edges=True)
        def make_edges(g):
            dmin = float(np.floor(g[depth_col].min()))
            dmax = float(np.ceil(g[depth_col].max()))
            edges = np.arange(dmin, dmax + bin_size, bin_size)
            return edges if edges.size >= 2 else None

        shared_edges = make_edges(g_cast) if not per_phase_edges else None
        if not per_phase_edges and shared_edges is None:
            continue

        # ---- Process each phase ----
        for ph, gp in g_cast.groupby("phase", sort=True):
            if gp.empty:
                continue

            edges = make_edges(gp) if per_phase_edges else shared_edges
            if edges is None:
                continue

            ivals = pd.IntervalIndex.from_breaks(edges, closed=closed)

            gg = gp.sort_values(time_col).copy()
            gg["_bin"] = pd.cut(gg[depth_col], bins=ivals)
            #
            # New aggregation logic
            # add code for aggregation_columns
            aggregation_columns = [col for col in gg.columns if col not in [cast_col, time_col, depth_col, "phase", "_bin"]]

            keywords=['fix_latitude', 'fix_longitude', 'salinity', 'temperature',
            'mass_concentration_of_oxygen_in_sea_water',
            'bin_mean_sea_water_salinity', 'bin_median_sea_water_salinity',
            'bin_mean_sea_water_temperature', 'bin_median_sea_water_temperature',
            'PAR', 'chlorophyll', 'chl', 'bbp470', 'bbp650',
            'volumescatcoeff117deg470nm', 'volumescatcoeff117deg650nm',
            'particulatebackscatteringcoeff470nm',
            'particulatebackscatteringcoeff650nm',
            'fix_residual_percent_distance_traveled_DeadReckonUsingMultipleVelocitySources',
            'latitude', 'longitude', 'depth', 'phase', 'cast_id', 'turning_point',
            'cast_id_unique', 'distance_km']
            # do we need other keywords?  Do we want to add rois_count and the species rates?
            
            agg_dict = {}
            for col in aggregation_columns:
                col_lower=col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = custom_sum_with_min_counter
                    #agg_dict[col] = 'sum'

            # Numeric means per observed bin
            # why do we use a mean here and not the sum?
            agg= gg.groupby("_bin", observed=True).agg(agg_dict)
            #
            # end of new code
            #
            # old code:
            #agg = gg.groupby("_bin", observed=True).mean(numeric_only=True)
            if agg.empty:
                continue

            # Bin center
            bin_center_col = f"{depth_col}_bin_center"
            agg[bin_center_col] = [iv.mid for iv in agg.index]
            agg[cast_col] = cid
            agg["phase"] = ph

            # Representative timestamp: nearest to bin center within phase subset
            rep_times = []
            for iv in agg.index:
                sub = gg.loc[gg["_bin"] == iv]
                if sub.empty:
                    rep_times.append(pd.NaT)
                    continue
                mid = iv.mid
                idx = (sub[depth_col] - mid).abs().idxmin()
                rep_times.append(gg.loc[idx, time_col])
            agg["rep_timestamp"] = pd.to_datetime(rep_times)

            # --- Time-in-bin: sum dt where next sample remains in the same bin ---
            gs = gg.copy()
            gs["__next_time__"] = gs[time_col].shift(-1)
            gs["__next_bin__"] = gs["_bin"].shift(-1)
            # original line:
            mask = gs["_bin"].notna() & (gs["_bin"] == gs["__next_bin__"])
            #mask = gs["_bin"] == gs["__next_bin__"]
            dt = (gs.loc[mask, "__next_time__"] - gs.loc[mask, time_col]).dt.total_seconds()
            elapsed = dt.groupby(gs.loc[mask, "_bin"], observed=True).sum(min_count=1)
            
            elapsed = elapsed.reindex(agg.index)  # align to bins present in agg

            # Start/end times per bin (first/last timestamps observed in that bin)
            ts_first = gg.groupby("_bin", observed=True)[time_col].first().reindex(agg.index)
            ts_last  = gg.groupby("_bin", observed=True)[time_col].last().reindex(agg.index)

            agg["start_time"] = pd.to_datetime(ts_first.values)
            agg["end_time"] = pd.to_datetime(ts_last.values)
            agg["elapsed_seconds"]=elapsed.astype(float)
            # Supposedly this line gets rid of all NaNs but we need to keep them and true zeros
            #agg["elapsed_seconds"] = elapsed.fillna(0.0).astype(float)

            out_frames.append(agg.reset_index(drop=True))

    if not out_frames:
        cols = [cast_col, "phase", f"{depth_col}_bin_center", "rep_timestamp", "start_time", "end_time", "elapsed_seconds"]
        return pd.DataFrame(columns=cols).set_index([cast_col, "phase", f"{depth_col}_bin_center"])

    result = (
        pd.concat(out_frames, ignore_index=True)
          .set_index([cast_col, "phase", f"{depth_col}_bin_center"])
          .sort_index()
    )

