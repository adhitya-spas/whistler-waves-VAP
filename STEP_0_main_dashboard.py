"""
Created on Wed Jan  7 07:21:58 2026
@author: Remote
"""

# IMPORTS
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from STEP_1_get_data import (
    check_available_files,
    read_magephem_h5,
    read_emfisis_spectral_merged_files,
    read_wfr_waveform_continuous_burst,
    read_emfisis_hfr_density_files,
)
from STEP_2_plot_data import (
    plot_emfsis_mlat_combined,
    plot_emfsis_waveform,
    plot_emfsis_waveform_broken_x,
    plot_spacecraft_locations_meridional,
    plot_magephem_day_context,
    _fmt_ts,
)

###############################################################################
# CONFIGURATION - All defines and parameters in one place
###############################################################################

# Paths & Data Source Configuration
ROOT_DIR = r"F:\UAF_Research"
MAGEPHEM_MODEL = "T89D" #"T89D" #"OP77Q" #"T89Q" #"TS04D"

# Time Range Configuration
START_DATETIME = "2012-10-31 00:00:00"
END_DATETIME = "2012-10-31 23:59:59"

# Feature Flags & Display Options
DEBUG_PLOT_CONTEXT = True               # show fce overlays, Kp/Dst, MagEphem context, Earth+L plots
PLOT_EXTRA_CONTEXT = True               # plot zoomed spectral, trajectory, and extended magephem context

# Plot Configuration (STEP_2)
RE_KM = 6371.0                          # Earth radius in km
PLOT_WAVEFORM_FIGSIZE = (9.6, 4.6)      # Waveform spectrogram figure size
PLOT_WAVEFORM_VMIN = -13                # colormap settings for waveform spectrograms
PLOT_WAVEFORM_VMAX = -7                 
PLOT_WAVEFORM_CMAP = "turbo"            
WAVEFORM_GAP_THRESHOLD_S = 20.0         # broken-x plot settings
WAVEFORM_MAX_PANELS = 20                

###############################################################################
# MAIN EXECUTION
###############################################################################

resA = check_available_files("RBSP-A", START_DATETIME, END_DATETIME, root_dir=ROOT_DIR, magephem_model=MAGEPHEM_MODEL)
resB = check_available_files("RBSP-B", START_DATETIME, END_DATETIME, root_dir=ROOT_DIR, magephem_model=MAGEPHEM_MODEL)

start_dt = datetime.strptime(START_DATETIME, "%Y-%m-%d %H:%M:%S")
end_dt = datetime.strptime(END_DATETIME, "%Y-%m-%d %H:%M:%S")
current_date = start_dt.date()

# Initialize data storage containers
magephem_dfs = {"RBSP-A": {}, "RBSP-B": {}}
emfisis_wfr_dfs = {"RBSP-A": {}, "RBSP-B": {}}
emfisis_freqs = {"RBSP-A": None, "RBSP-B": None}
waveform_data = {"RBSP-A": {}, "RBSP-B": {}}
hfr_dfs       = {"RBSP-A": {}, "RBSP-B": {}}  # HFR electron density per day

###############################################################################
# SECTION 1: Load data from files (MagEphem, EMFISIS spectral, waveforms)
###############################################################################

while current_date <= end_dt.date():
    date_str = f"{current_date:%Y-%m-%d}"
    
    for sc, res in [("RBSP-A", resA), ("RBSP-B", resB)]:
        # 1a. Load MagEphem ephemeris and field model data
        if date_str in res["magephem"]:
            entry = res["magephem"][date_str]
            if entry["h5"]:
                print(f"\nLoading {sc} MagEphem for {date_str}...")
                df = read_magephem_h5(entry["h5"])
                if df is not None and len(df) > 0:
                    magephem_dfs[sc][date_str] = df
                    print(f"Loaded: shape={df.shape}, columns={len(df.columns)}")
                    key_cols = [c for c in ["IsoTime", "Rsm_0", "Rsm_1", "Rsm_2", 
                                            "CDMAG_MLAT", "CDMAG_MLT", "Lsimple"] if c in df.columns]
                    if key_cols:
                        print(f"Key columns available: {', '.join(key_cols)}")
        
        # 1b. Load EMFISIS spectral data (frequency content of magnetic field)
        files = res["emfisis_spectral"].get(date_str, [])
        spectral_merged_files = [p for p in files if ("diagonal-merged" in p.name and p.suffix.lower() == ".cdf")]
        spectral_diag_files = [p for p in files if ("spectral-matrix-diagonal" in p.name and "merged" not in p.name and p.suffix.lower() == ".cdf")]
        
        spectral_to_try = spectral_merged_files if spectral_merged_files else spectral_diag_files
        if spectral_to_try:
            try:
                t_spec, freq, data = read_emfisis_spectral_merged_files(spectral_to_try)
                print(f"Spectral-merged ({sc}) {date_str}: times shape={len(t_spec)}, freqs shape={freq.shape}, data shape={data.shape}")
                df_spec = pd.DataFrame(data, index=t_spec, columns=[f"freq_{i}" for i in range(data.shape[1])])
                df_spec.index.name = "Time"
                emfisis_wfr_dfs[sc][date_str] = df_spec
                if emfisis_freqs[sc] is None:
                    emfisis_freqs[sc] = freq
            except Exception as e:
                # If merged failed and diagonal exists, try diagonal
                if spectral_merged_files and spectral_diag_files:
                    try:
                        t_spec, freq, data = read_emfisis_spectral_merged_files(spectral_diag_files)
                        print(f"Spectral-diagonal ({sc}) {date_str}: times shape={len(t_spec)}, freqs shape={freq.shape}, data shape={data.shape}")
                        df_spec = pd.DataFrame(data, index=t_spec, columns=[f"freq_{i}" for i in range(data.shape[1])])
                        df_spec.index.name = "Time"
                        emfisis_wfr_dfs[sc][date_str] = df_spec
                        if emfisis_freqs[sc] is None:
                            emfisis_freqs[sc] = freq
                    except Exception as e2:
                        print(f"Spectral parse ({sc}) {date_str}: {e2}")
                else:
                    print(f"Spectral parse ({sc}) {date_str}: {e}")
        
        # 1c. Load EMFISIS waveform burst data (raw time series at high sampling rate)
        wave_files = res.get("emfisis_waveform", {}).get(date_str, [])
        if wave_files:
            for wf_path in wave_files:
                try:
                    t_rec, wf_rec, fs = read_wfr_waveform_continuous_burst(wf_path, component="Bw")
                    mage_day = magephem_dfs.get(sc, {}).get(date_str)
                    event_utc = pd.to_datetime(t_rec[0]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    # Print record info
                    wf_arr = np.asarray(wf_rec)
                    nrec = wf_arr.shape[0]
                    nsamp = wf_arr.shape[1]
                    print(f"  Nrecords: {nrec}, Nsamp/rec: {nsamp}, fs: {fs} Hz")
                    
                    if date_str not in waveform_data[sc]:
                        waveform_data[sc][date_str] = []
                    waveform_data[sc][date_str].append((t_rec, wf_rec, fs, event_utc, mage_day, wf_path))
                    print(f"Loaded waveform ({sc}) {date_str}: {wf_path.name}")
                except Exception as e:
                    print(f"[WARN] {sc}: waveform load skipped for {wf_path.name}: {e}")

        # 1d. Load EMFISIS HFR electron density (optional overlay on f_ce panel)
        # Actual filenames: *HFR-spectra-merged*.cdf (preferred) or *HFR-spectra*.cdf
        l2_folder = Path(ROOT_DIR) / sc / "L2" / f"{current_date:%Y}" / f"{current_date:%m}" / f"{current_date:%d}"
        hfr_merged = sorted(l2_folder.glob("*HFR-spectra-merged*.cdf"))
        hfr_plain  = sorted(l2_folder.glob("*HFR-spectra*.cdf"))
        hfr_plain  = [p for p in hfr_plain if "merged" not in p.name]  # exclude merged from fallback list
        hfr_files  = hfr_merged if hfr_merged else hfr_plain
        if hfr_files:
            print(f"\nLoading {sc} HFR density for {date_str} ({len(hfr_files)} file(s))...")
            try:
                hfr_day = read_emfisis_hfr_density_files(hfr_files)
                if hfr_day is not None and len(hfr_day) > 0:
                    hfr_dfs[sc][date_str] = hfr_day
                    print(f"  Loaded HFR density ({sc}) {date_str}: {len(hfr_day)} samples, Ne range [{hfr_day['Ne_cm3'].min():.2g}, {hfr_day['Ne_cm3'].max():.2g}] cm^-3")
                else:
                    print(f"  [WARN] HFR density ({sc}) {date_str}: empty after read")
            except Exception as e:
                print(f"  [WARN] HFR density ({sc}) {date_str}: {e}")
        else:
            print(f"  HFR density ({sc}) {date_str}: no files found in {l2_folder}")
        
    
    current_date += timedelta(days=1)

###############################################################################
# Load Summary
###############################################################################
print("\n" + "=" * 80)
print("Load Summary")
print(f"RBSP-A: Loaded {len(magephem_dfs['RBSP-A'])} days of MagEphem data")
print(f"RBSP-A: Loaded {len(emfisis_wfr_dfs['RBSP-A'])} days of spectral-merged data")
print(f"RBSP-A: Loaded {len(waveform_data['RBSP-A'])} days of spectral waveform data with {len([item for sublist in waveform_data['RBSP-A'].values() for item in sublist])} files")
print(f"RBSP-A: Loaded {len(hfr_dfs['RBSP-A'])} days of HFR density data")
print(f"RBSP-B: Loaded {len(magephem_dfs['RBSP-B'])} days of MagEphem data")
print(f"RBSP-B: Loaded {len(emfisis_wfr_dfs['RBSP-B'])} days of spectral-merged data")
print(f"RBSP-B: Loaded {len(waveform_data['RBSP-B'])} days of spectral waveform data with {len([item for sublist in waveform_data['RBSP-B'].values() for item in sublist])} files")
print(f"RBSP-B: Loaded {len(hfr_dfs['RBSP-B'])} days of HFR density data")
print("=" * 80)

###############################################################################
# SECTION 2: Plot EMFISIS spectrograms and waveforms
###############################################################################

for sc, res in [("RBSP-A", resA), ("RBSP-B", resB)]:
    # Plot 2a: EMFISIS frequency spectrogram (full frequency content over full time range)
    if emfisis_wfr_dfs[sc] and emfisis_freqs[sc] is not None:
        combined = pd.concat(emfisis_wfr_dfs[sc].values()).sort_index()
        mage_df = pd.concat(magephem_dfs[sc].values(), ignore_index=True) if magephem_dfs[sc] else None
        
        t_start = combined.index[0].strftime("%Y-%m-%d %H:%M:%S")
        t_end = combined.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        date_fmt = combined.index[0].strftime("%Y%m%d")

        # Collect burst event times and intervals across all waveform files for this spacecraft
        burst_times_list = []
        burst_intervals_list = []
        for _date_key, _entries in waveform_data[sc].items():
            for _t_rec, _wf_rec, _fs, _event_utc, _, _wf_path in _entries:
                _event_dt = pd.to_datetime(_event_utc, utc=True, errors="coerce")
                if not pd.isna(_event_dt):
                    burst_times_list.append(_event_dt)
                # Derive start/end from t_rec for shaded regions
                _t_arr = pd.to_datetime(_t_rec, utc=True, errors="coerce")
                _valid = _t_arr[~pd.isna(_t_arr)]
                if len(_valid) > 0:
                    burst_intervals_list.append((_valid[0], _valid[-1]))

        spec_outpath = Path(ROOT_DIR) / "plots" / f"{sc}_{date_fmt}_emfisis_com.png"
        spec_outpath.parent.mkdir(parents=True, exist_ok=True)

        # Combine HFR density across days for this spacecraft
        hfr_combined = None
        if hfr_dfs[sc]:
            hfr_combined = pd.concat(hfr_dfs[sc].values()).sort_index()
            hfr_combined = hfr_combined[~hfr_combined.index.duplicated(keep="first")]
            print(f"  HFR density ({sc}): {len(hfr_combined)} total samples available for f_pe overlay")

        if mage_df is not None:
            plot_emfsis_mlat_combined(
                spectral_times=combined.index,
                spectral_freqs_hz=emfisis_freqs[sc],
                spectral_log10=combined.values,
                magephem_df=mage_df,
                t_start_utc=t_start,
                t_end_utc=t_end,
                spacecraft=sc,
                burst_times=burst_times_list if burst_times_list else None,
                burst_intervals=burst_intervals_list if burst_intervals_list else None,
                hfr_df=hfr_combined,
                outpath=str(spec_outpath),
            )
            print(f"[PLOTTED] {sc} EMFISIS Combined: {t_start} to {t_end}")
        else:
            print(f"[WARN] {sc}: no MagEphem data, skipping combined plot")
        
        # Plot day-long MagEphem context (once per spacecraft/day)
        if mage_df is not None and DEBUG_PLOT_CONTEXT:
            date_str = combined.index[0].strftime("%Y-%m-%d")
            outdir = Path(ROOT_DIR) / "plots"
            outdir.mkdir(parents=True, exist_ok=True)
            plot_magephem_day_context(
                spacecraft=sc,
                date_str=date_str,
                mage_df=mage_df,
                t_start=t_start,
                t_end=t_end,
                out_dir=str(outdir),
            )

    else:
        print(f"[INFO] {sc}: no spectral data to plot")

    # Plot 2b & 2c: EMFISIS waveform spectrograms (high-resolution burst data)
    if not waveform_data[sc]:
        print(f"[INFO] {sc}: no waveform files in range")
    else:
        wf_counter = 0
        for date_str in sorted(waveform_data[sc].keys()):
            for t_rec, wf_rec, fs, event_utc, mage_day, wf_path in waveform_data[sc][date_str]:
                wf_counter += 1
                try:
                    outdir = Path(wf_path).parent
                    event_dt = pd.to_datetime(event_utc, utc=True, errors="coerce")
                    
                    plot_emfsis_waveform(
                        t_rec,
                        wf_rec,
                        fs,
                        spacecraft=sc,
                        event_utc=event_utc,
                        record_number=1,
                        magephem_df=mage_day,
                        component="Bw",
                        figsize=PLOT_WAVEFORM_FIGSIZE,
                        vmin=PLOT_WAVEFORM_VMIN,
                        vmax=PLOT_WAVEFORM_VMAX,
                        cmap=PLOT_WAVEFORM_CMAP,
                        show_fce=DEBUG_PLOT_CONTEXT,
                        outpath=str(outdir / f"{sc}_{_fmt_ts(event_dt)}_waveform.png") if event_dt is not None and not pd.isna(event_dt) else None,
                    )
                    print(f"[PLOTTED] {sc} EMFISIS Waveform: {wf_counter} at {event_utc}")
                    
                    # Broken-x waveform plot with proper filename
                    t_rec_pd = pd.to_datetime(t_rec, utc=True)
                    if len(t_rec_pd) > 0:
                        t_start_rec = t_rec_pd[0]
                        t_end_rec = t_rec_pd[-1]
                        brokenx_path = outdir / f"{sc}_{_fmt_ts(t_start_rec)}_{_fmt_ts(t_end_rec)}_waveform_brokenx.png"
                    elif event_dt is not None and not pd.isna(event_dt):
                        brokenx_path = outdir / f"{sc}_{_fmt_ts(event_dt)}_waveform_brokenx.png"
                    else:
                        brokenx_path = None
                    
                    plot_emfsis_waveform_broken_x(
                        times_records=t_rec,
                        waveform_2d=wf_rec,
                        fs=fs,
                        spacecraft=sc,
                        component="Bw",
                        gap_threshold_s=WAVEFORM_GAP_THRESHOLD_S,
                        max_panels=WAVEFORM_MAX_PANELS,
                        outpath=str(brokenx_path) if brokenx_path else None,
                    )
                    print(f"[PLOTTED] {sc} EMFISIS Waveform (All Records): at {event_utc}")

                except Exception as e:
                    print(f"[WARN] {sc}: waveform plot skipped for {wf_path.name}")
                    print(f"  Error: {e}")
                    print(f"  Debug: t_rec shape={np.asarray(t_rec).shape}, wf_rec shape={np.asarray(wf_rec).shape}")
                    print(f"  File: {wf_path.name}, Event UTC: {event_utc}")

###############################################################################
# SECTION 3: Plot spacecraft locations (meridional plane)
###############################################################################

for sc in ["RBSP-A", "RBSP-B"]:
    if magephem_dfs[sc]:
        mage_combined = pd.concat(magephem_dfs[sc].values(), ignore_index=True)
        
        try:
            plot_spacecraft_locations_meridional(
                mage_combined,
                start_utc=START_DATETIME,
                end_utc=END_DATETIME,
                spacecraft=sc,
            )
            print(f"[PLOTTED] {sc} Spacecraft Locations (Meridional) from {START_DATETIME} to {END_DATETIME}")
        except Exception as e:
            print(f"[WARN] {sc}: spacecraft location plot failed: {e}")
    else:
        print(f"[INFO] {sc}: no MagEphem data available for location plot")
