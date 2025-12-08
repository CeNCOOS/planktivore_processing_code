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
rom haversine import haversine
import matplotlib.dates as mdates

import planktivore_temporal_avg_modules
#
# Bin planktivore data by depth
#
# config we want to pass or setup somehow
fig_dir=Path('')
data_dir=Path('')
image_dir=Path('')
camlog_dir=Path('')
camlog_ext=[]
input_filenames_list=[]
input_inference_filename=[]
output_merged_dataframe=[]
parquet_filenames_file=[]
#
# Read the image filenames and compute times from the names
#
df=read_image_names(image_dir,parquet_filenames_file)
#
# read the inference data
#
inference = pd.read_parquet(input_inference_filename)
# Copied from config.json
id2label = {
    "0": "Akashiwo",
    "1": "Amphidinium_Oxyphysis",
    "2": "Ceratium",
    "3": "Chaetoceros",
    "4": "Ciliate",
    "5": "Cylindrotheca",
    "6": "Detonula_Cerataulina_Lauderia",
    "7": "Detritus",
    "8": "Dictyocha",
    "9": "Dinoflagellate",
    "10": "Eucampia",
    "11": "Guinardia_Dactyliosolen",
    "12": "Gyrodinium",
    "13": "Medium_pennate",
    "14": "Mesodinium",
    "15": "Mixed_diatom_chain",
    "16": "Nano_plankton",
    "17": "Polykrikos",
    "18": "Prorocentrum",
    "19": "Pseudo-nitzschia",
    "20": "Strombidium",
    "21": "Thalassionema",
    "22": "Thalassiosira",
    "23": "Tiarina",
    "24": "Truncated",
    "25": "unknown_flagellate"
}
all_labels = list(id2label.values())
#inference['label1'].unique()
# compute times from filenames
inference['fileName'] = inference['file'].str.split("/", expand=True).iloc[:, -1]
inference['dateStr'] = inference["fileName"].str.split("-",expand=True)[1]
numeric_microseconds=pd.to_numeric(inference['dateStr'])
inference['dateTime'] = pd.to_datetime(numeric_microseconds, unit='us')
inference['dateTime'] = inference['dateTime'].dt.tz_localize('UTC').dt.tz_localize(None)
inference.sort_values('dateTime', inplace=True)
# new code to drop values which are not confident
# actually we want to rename these values 
# add a boolean column for confident or not
# this avoids having to name the untrusted values with bad_image or other
# where other is specifically used for non-living things (i.e. IFCB PTWG, Standard and practices for reporting plankton and other particle observations from images, 2020)
conf_threshold = 0.5
inference_filtered=inference.copy()
inference_filtered['trust'] = inference_filtered['score1'] >= conf_threshold
#
# load the physical data from the vehicle
#
lrauv_fnames=sorted(glob.glob(data_dir+"*_sciencg.nc"))
df=xr.open_mfdataset(lrauv_fnames,
                     join='outer',
                     compat='no_conflicts')
# remove unneeded variables
ds=ds.drop_vars('health_platform_battery_charge')
ds=ds.drop_vars('health_platform_average_voltage')
ds=ds.drop_vars('health_platform_average_current')
ds=ds.drop_vars('control_inputs_buoyancy_position')
ds=ds.drop_vars('control_inputs_elevator_angle')
ds=ds.drop_vars('control_inputs_rudder_angle')
ds=ds.drop_vars('control_inputs_mass_position')
ds=ds.drop_vars('pose_latitude_DeadReckonUsingMultipleVelocitySources')
ds=ds.drop_vars('pose_longitude_DeadReckonUsingMultipleVelocitySources')
ds=ds.drop_vars('pose_depth_DeadReckonUsingMultipleVelocitySources')
ds=ds.drop_vars('pitch')
ds=ds.drop_vars('roll')
ds=ds.drop_vars('yaw')
# make flat dataframe from profile data
lrauv_raw = ds.to_dataframe()
profiles = lrauv_raw.copy()
# This code allows for automation of removing flat profiles
# this is before profiles are defined.  Problem is that it leaves the very first point of a flat region.
flatvalues=find_flat_line_regions(lrauv_raw,'depth',300,14,0.5)
profiles=lrauv_raw.loc[flatvalues]
#
# find the cast turning points
#
labeled = label_casts_by_turning_points(
    profiles,
    depth_col="depth",
    smooth=True,
    win_len=21,
    polyorder=2,
    peak_prominence=4,  # raise to skip tiny kinks; lower if casts are gentle
    peak_distance=20       # increase if you’re over-detecting turns
)
#
# create cast id
#
mask = labeled['cast_id'].notna()
labeled.loc[mask, 'cast_id_unique'] = (
    labeled.loc[mask, 'cast_id'] * 2 +
    (labeled.loc[mask, 'phase'] == 'up').astype(int) - 1
).astype('Int64')
#
# do some salinity QC
#
# Now we can clean salinity based on jumps
# This is not perfect but should help
salty=labeled['salinity']
bads=np.where(np.abs(np.diff(salty))> 0.25)[0]
badlabels=labeled.index[bads+1]
labeled.loc[badlabels,'salinity']=np.nan
#
# clean up casts based upon Synchro need for profiles and to remove interpolated data
#
x=np.unique(labeled['cast_id'][:])
ncasts=np.arange(0,len(x))
WINDOW=15
MIN_R2=0.999
new_labeled=labeled.copy()
exclude_columns=['cast_id','depth','phase','turning_point','cast_id_unique','time','latitude','longitude','fix_latitude','fix_longitude','fix_residual_percent_distance_traveled_DeadReckonUsingMultipleVelocitySources']
all_labels_clean=[col for col in labeled.columns if col not in exclude_columns]
for i in ncasts:
    castdown=(labeled['cast_id']==x[i])&(labeled['phase']=='down')
    full_indexd=castdown.index
    castup=(labeled['cast_id']==x[i])&(labeled['phase']=='up')
    full_indexu=castup.index
    interpolated_maskd = detect_interpolated_segments_rolling(labeled[castdown], 'temperature', 'depth', WINDOW, MIN_R2)
    interpolated_masku = detect_interpolated_segments_rolling(labeled[castup], 'temperature', 'depth', WINDOW, MIN_R2)
    reindex_down = interpolated_maskd.reindex(full_indexd, fill_value=False)
    reindex_up = interpolated_masku.reindex(full_indexu, fill_value=False)
    castdown_bool=castdown.astype(bool)
    castup_bool=castup.astype(bool)
    reindex_down_bool=reindex_down.astype(bool)
    reindex_up_bool=reindex_up.astype(bool)
    final_clean_down_condition=castdown_bool & reindex_down_bool
    final_clean_up_condition=castup_bool & reindex_up_bool
    #final_clean_down_condition=castdown & interpolated_maskd
    #final_clean_up_condition=castup & interpolated_masku
    new_labeled.loc[final_clean_down_condition,all_labels_clean]=np.nan
    new_labeled.loc[final_clean_up_condition,all_labels_clean]=np.nan
#
# compute distance and add to dataframe
#
distances = []
for lat,lon in labeled[['latitude','longitude']].itertuples(index=False):
    dist = haversine((36.7783, -121.85), (lat, lon), unit='km')
    distances.append(dist)
labeled['distance_km'] = distances
#
# read the camera log files
#
camdf=read_planktivore_camlogs(camlog_dir,camlog_ext)
camdf['dateTime']=pd.to_datetime(camdf[['year','month','day','hours','minute','second']])
#
# Add the ROIS counts per sample and using the camera logs add zeros vs NaNs.
#
edges = pd.to_datetime(labeled.index).sort_values().unique()
bins = pd.IntervalIndex.from_breaks(edges, closed="left")
bin_codes = pd.cut(df["dateTime"], bins=bins, right=False)
bin_code_cam=pd.cut(camdf["dateTime"], bins=bins, right=False)
counts=df.groupby(bin_codes,observed=False).size()
cam_counts=camdf.groupby(bin_code_cam,observed=False).size()
# alternative to compute for other column information
cam_groups=camdf.groupby(bin_code_cam,observed=False)
columns_to_sum=['highmag_detections (#/s)','highmag_rois (#/s)','highmag_saved_rois (#/s)','highmag_average_area (pixels)','lowmag_detections (#/s)','lowmag_rois (#/s)','lowmag_saved_rois (#/s)','lowmag_average_area (pixels)']
cam_sums=cam_groups[columns_to_sum].sum()
#
# I think this is were we put NaNs in for no data from the cam_counts
cam_counts = cam_counts.reindex(bins, fill_value=np.nan)
ll=cam_counts.values==0
cam_counts[ll]=np.nan
counts[ll]=np.nan
for col in columns_to_sum:
    cam_sums[col] = cam_sums[col].reindex(bins, fill_value=np.nan)
    cam_sums.loc[ll, col] = np.nan
cam_sums.index=cam_sums.index.categories.left
cam_counts.index = cam_counts.index.left
counts.index = counts.index.categories.left
labeled["rois_count"] = counts.reindex(labeled.index, fill_value=np.nan).astype('Int64')
#
labeled['highmag_detections'] = cam_sums['highmag_detections (#/s)'].reindex(labeled.index, fill_value=np.nan)
labeled['highmag_rois'] = cam_sums['highmag_rois (#/s)'].reindex(labeled.index, fill_value=np.nan)
labeled['highmag_saved_rois'] = cam_sums['highmag_saved_rois (#/s)'].reindex(labeled.index, fill_value=np.nan)
labeled['highmag_average_area'] = cam_sums['highmag_average_area (pixels)'].reindex(labeled.index, fill_value=np.nan)
#
# Compute elapsed time and
#
elapsed_seconds = (
    labeled.index.to_series()
    .diff()
    .dt.total_seconds()
    .shift(-1)        # shift so first interval is at position 0
    .astype('Int64')
)

# Timedelta differences as integer seconds (nullable Int64 keeps the first NA)
labeled['rois_norm'] = labeled['rois_count'] / elapsed_seconds
#
# Add inference data to the profiles
#
# Your target labels
labels = [
 'Akashiwo',
 'Amphidinium_Oxyphysis',
 'Ceratium',
 'Chaetoceros',
 'Ciliate',
 'Cylindrotheca',
 'Detonula_Cerataulina_Lauderia',
 'Detritus',
 'Dictyocha',
 'Dinoflagellate',
 'Eucampia',
 'Guinardia_Dactyliosolen',
 'Gyrodinium',
 'Medium_pennate',
 'Mesodinium',
 'Mixed_diatom_chain',
 'Nano_plankton',
 'Polykrikos',
 'Prorocentrum',
 'Pseudo-nitzschia',
 'Strombidium',
 'Thalassionema',
 'Thalassiosira',
 'Tiarina',
 'Truncated',
 'unknown_flagellate'
]

# Suppose your dataframe has a 'label' column
# Example: counts = df['label'].value_counts()
#counts = inference["label1"].value_counts()
counts = inference_filtered["label1"].value_counts()

# Reindex with your full list and fill missing with zero
counts = counts.reindex(labels, fill_value=0)

# Optionally make it a DataFrame
label_counts = counts.reset_index()
label_counts.columns = ["label", "count"]

# --- Assumptions ---
# labeled.index: strictly increasing, unique DatetimeIndex (bin edges)
# inference: columns ['dateTime', 'label1'] ; dateTime is datetime64
# all_labels: list of label names (columns order you want)

# 0) Ensure types
edges = pd.to_datetime(labeled.index).sort_values().unique()
# original code
#inference = inference.copy()
#inference["dateTime"] = pd.to_datetime(inference["dateTime"])
#inference["label1"] = pd.Categorical(inference["label1"], categories=all_labels)
inference_filtered = inference_filtered.copy()
inference_filtered["dateTime"] = pd.to_datetime(inference_filtered["dateTime"])
inference_filtered["label1"] = pd.Categorical(inference_filtered["label1"], categories=all_labels)

# 1) Build left-closed, right-open bins: [start, end)
bins = pd.IntervalIndex.from_breaks(edges, closed="left")

# 2) Bin events
#binned = pd.cut(inference["dateTime"], bins=bins, right=False) # original code
binned = pd.cut(inference_filtered["dateTime"], bins=bins, right=False)

# 3) Count per interval × label (use observed=True to avoid the FutureWarning)
# original code
#counts = (
#    inference.groupby([binned, "label1"], observed=True)
#    .size()
#    .unstack("label1", fill_value=0)
#    .reindex(columns=all_labels, fill_value=0)
#)
counts = (
    inference_filtered.groupby([binned, "label1"], observed=True)
    .size()
    .unstack("label1", fill_value=0)
    .reindex(columns=all_labels, fill_value=0)
)

# 4) Ensure **every** interval is present (even empty ones)
counts = counts.reindex(bins, fill_value=0)

# 5) Replace IntervalIndex with its left edge (DatetimeIndex)
counts.index = counts.index.left  # -> Interval start time

# 6) Compute elapsed seconds for each interval (aligned to left edge)
dt = pd.Series(edges, index=edges)            # helpful alignment
elapsed_seconds = (dt.shift(-1) - dt).dt.total_seconds()
elapsed_seconds = elapsed_seconds.loc[counts.index]  # align to left edges

# 7) Rates (counts per second): last bin has NaN duration → fill with 0 later
rates = counts.div(elapsed_seconds, axis=0)

# 8) Join into labeled by left edge, zero-fill where missing
# (This preserves your original labeled rows, including the last edge)
labeled = labeled.join(counts, how="left", rsuffix="_count")
# Do we really want to fill with 0s here?  Maybe NaN is better to indicate no data?
# We are now preserving NaNs and 0s we are not "zero-filling"
#labeled = labeled.fillna(0)

# 9) If you also want the rates joined (optional):
rate_cols = [f"{c}_rate" for c in counts.columns]
rates.columns = rate_cols
# Again not sure we want to fill in with 0s here.  Maybe NaN is better.
# Keeping NaN and 0 as a NaN is when there truely was no data (camera didn't record, vs camera was on
# but no significant ROIs were written.
# Checkpoint
labeled.to_parquet("c:/planktivore/lrauv_april2025_with_class_20251009model_filtered_addNaN.parquet")
#
# Now onto the depth binning
#
df=labeled.copy()
ll=df['cast_id']==1
cast_col='cast_id'
depth_col='depth'
time_col=None
bin_size=5
smooth_window=5
phase_labels=["down", "up"]
closed='right'
per_phase_edges: bool = False
work = df.copy()
work=work[ll]
def custom_sum_with_min_counter(series):
    return series.sum(min_count=1)
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
    out_frames = []

# ---- Process each cast ----
for cid, g_cast in work.groupby(cast_col, sort=True):
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
    g_cast = g_cast.sort_values(time_col).copy()
    # Direction: positive depth change => down; negative => up
    ddepth = g_cast[depth_col].diff()
    if smooth_window and smooth_window > 1:
        ddepth = ddepth.rolling(smooth_window, center=True, min_periods=1).median()
        sign = np.sign(ddepth).replace({0: np.nan})
        sign = sign.ffill().bfill()
        phase_map = {1.0: phase_labels[0], -1.0: phase_labels[1]}
        g_cast["phase"] = sign.map(phase_map)
    def make_edges(g):
        dmin = float(np.floor(g[depth_col].min()))
        dmax = float(np.ceil(g[depth_col].max()))
        edges = np.arange(dmin, dmax + bin_size, bin_size)
        return edges if edges.size >= 2 else None

    shared_edges = make_edges(g_cast) if not per_phase_edges else None
    if not per_phase_edges and shared_edges is None:
        continue
    for ph, gp in g_cast.groupby("phase", sort=True):
        if gp.empty:
            continue

        edges = make_edges(gp) if per_phase_edges else shared_edges
        if edges is None:
            continue

        ivals = pd.IntervalIndex.from_breaks(edges, closed=closed)

        gg = gp.sort_values(time_col).copy()
        # new force data to be float
        #gg=gg.astype('float64')
        #
        gg["_bin"] = pd.cut(gg[depth_col], bins=ivals)
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
                #agg_dict[col] = 'sum'
                agg_dict[col] = custom_sum_with_min_counter
    
        agg= gg.groupby("_bin", observed=True).agg(agg_dict)
        #
        # end of new code
        #
        # old code:
        ##agg = gg.groupby("_bin", observed=True).mean(numeric_only=True)
        #if agg.empty:
        #    continue

        ## Bin center
        #bin_center_col = f"{depth_col}_bin_center"
        #agg[bin_center_col] = [iv.mid for iv in agg.index]
        #agg[cast_col] = cid
        #agg["phase"] = ph

binned = bin_casts_avg_with_time_updown(
    df.reset_index(drop=False),
    depth_col="depth",
    cast_col="cast_id",
    time_col="time",   # or None to use datetime index
    bin_size=5,
    phase_labels=("down", "up"),
    smooth_window=5,
#    smooth_window=17,
    closed="right",
    per_phase_edges=True,  # True if you want separate edges for up/down
)
binned = binned.reset_index()
binned['sigma_theta'] = gsw.sigma0(binned['salinity'], binned['temperature'])

nan_mask=binned['rois_count'].isna()
target_columns=['rois_count', 'rois_norm', 'Akashiwo',
       'Amphidinium_Oxyphysis', 'Ceratium', 'Chaetoceros', 'Ciliate',
       'Cylindrotheca', 'Detonula_Cerataulina_Lauderia', 'Detritus',
       'Dictyocha', 'Dinoflagellate', 'Eucampia', 'Guinardia_Dactyliosolen',
       'Gyrodinium', 'Medium_pennate', 'Mesodinium', 'Mixed_diatom_chain',
       'Nano_plankton', 'Polykrikos', 'Prorocentrum', 'Pseudo-nitzschia',
       'Strombidium', 'Thalassionema', 'Thalassiosira', 'Tiarina', 'Truncated',
       'unknown_flagellate']
binned.loc[nan_mask, target_columns]=np.nan
#
# write out a file
#
binned.to_csv(p / "lrauv_april2025_binned_depthtime_updown_5m_perphase_edges.csv", index=False)

