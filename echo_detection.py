# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:02:32 2026

@author: Remote
"""

# IMPORTS
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    plot_spacecraft_locations_meridional_fig,
    plot_magephem_day_context,
    _fmt_ts,
)

from echo_detection_helper import total_field_psd, means_wna_ellipticity_from_waveform, plot_waveform_and_filtered

###############################################################################
# CONFIGURATION - All defines and parameters in one place
###############################################################################

# Paths & Data Source Configuration
ROOT_DIR = r"F:\UAF_Research"
MAGEPHEM_MODEL = "T89D" #"T89D" #"OP77Q" #"T89Q" #"TS04D"

# Time Range Configuration
START_DATETIME = "2012-10-31 00:00:00"#"2012-10-07 00:00:00"
END_DATETIME = "2012-10-31 23:59:59"#"2012-10-07 23:59:59"

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
emfisis_freqs: dict[str, np.ndarray | None] = {"RBSP-A": None, "RBSP-B": None}
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
        
        # # 1b. Load EMFISIS spectral data (frequency content of magnetic field)
        # files = res["emfisis_spectral"].get(date_str, [])
        # spectral_merged_files = [p for p in files if ("diagonal-merged" in p.name and p.suffix.lower() == ".cdf")]
        # spectral_diag_files = [p for p in files if ("spectral-matrix-diagonal" in p.name and "merged" not in p.name and p.suffix.lower() == ".cdf")]
        
        # spectral_to_try = spectral_merged_files if spectral_merged_files else spectral_diag_files
        # if spectral_to_try:
        #     try:
        #         t_spec, freq, data = read_emfisis_spectral_merged_files(spectral_to_try)
        #         print(f"Spectral-merged ({sc}) {date_str}: times shape={len(t_spec)}, freqs shape={freq.shape}, data shape={data.shape}")
        #         df_spec = pd.DataFrame(data, index=t_spec, columns=[f"freq_{i}" for i in range(data.shape[1])])
        #         df_spec.index.name = "Time"
        #         emfisis_wfr_dfs[sc][date_str] = df_spec
        #         if emfisis_freqs[sc] is None:
        #             emfisis_freqs[sc] = freq
        #     except Exception as e:
        #         # If merged failed and diagonal exists, try diagonal
        #         if spectral_merged_files and spectral_diag_files:
        #             try:
        #                 t_spec, freq, data = read_emfisis_spectral_merged_files(spectral_diag_files)
        #                 print(f"Spectral-diagonal ({sc}) {date_str}: times shape={len(t_spec)}, freqs shape={freq.shape}, data shape={data.shape}")
        #                 df_spec = pd.DataFrame(data, index=t_spec, columns=[f"freq_{i}" for i in range(data.shape[1])])
        #                 df_spec.index.name = "Time"
        #                 emfisis_wfr_dfs[sc][date_str] = df_spec
        #                 if emfisis_freqs[sc] is None:
        #                     emfisis_freqs[sc] = freq
        #             except Exception as e2:
        #                 print(f"Spectral parse ({sc}) {date_str}: {e2}")
        #         else:
        #             print(f"Spectral parse ({sc}) {date_str}: {e}")
        
        # 1c. Load EMFISIS waveform burst data (raw time series at high sampling rate)
        wave_files = res.get("emfisis_waveform", {}).get(date_str, [])
        if wave_files:
            for wf_path in wave_files:
                try:
                    t_rec, wf_bu, fs = read_wfr_waveform_continuous_burst(wf_path, component="Bu")
                    _, wf_bv, _ = read_wfr_waveform_continuous_burst(wf_path, component="Bv")
                    _, wf_bw, _ = read_wfr_waveform_continuous_burst(wf_path, component="Bw")
                    _, wf_eu, _ = read_wfr_waveform_continuous_burst(wf_path, component="Eu")
                    _, wf_ev, _ = read_wfr_waveform_continuous_burst(wf_path, component="Ev")
                    _, wf_ew, _ = read_wfr_waveform_continuous_burst(wf_path, component="Ew")
                    mage_day = magephem_dfs.get(sc, {}).get(date_str)
                    event_utc = pd.to_datetime(t_rec[0]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    # Print record info
                    # wf_arr = np.asarray(wf_rec)
                    # nrec = wf_arr.shape[0]
                    # nsamp = wf_arr.shape[1]
                    
                    if date_str not in waveform_data[sc]:
                        waveform_data[sc][date_str] = []
                    waveform_data[sc][date_str].append((t_rec, wf_bu, wf_bv, wf_bw, wf_eu, wf_ev, wf_ew, fs, event_utc, mage_day, wf_path))
                    print(f"Loaded waveform ({sc}) {date_str}: {wf_path.name}")
                except Exception as e:
                    print(f"[WARN] {sc}: waveform load skipped for {wf_path.name}: {e}")

        # # 1d. Load EMFISIS HFR electron density (optional overlay on f_ce panel)
        # # Actual filenames: *HFR-spectra-merged*.cdf (preferred) or *HFR-spectra*.cdf
        # l2_folder = Path(ROOT_DIR) / sc / "L2" / f"{current_date:%Y}" / f"{current_date:%m}" / f"{current_date:%d}"
        # hfr_merged = sorted(l2_folder.glob("*HFR-spectra-merged*.cdf"))
        # hfr_plain  = sorted(p for p in l2_folder.glob("*HFR-spectra*.cdf") if "merged" not in p.name)
        # hfr_all    = list(dict.fromkeys(hfr_merged + hfr_plain))  # merged first, deduplicated

        # if hfr_all:
        #     print(f"\nLoading {sc} HFR density for {date_str} ({len(hfr_all)} file(s))...")
        #     hfr_day   = None
        #     remaining = list(hfr_all)
        #     while remaining:
        #         try:
        #             hfr_day = read_emfisis_hfr_density_files(remaining)
        #         except Exception as e:
        #             print(f"  [WARN] HFR density ({sc}) {date_str}: {e}")
        #             hfr_day = None
        #         if hfr_day is not None and len(hfr_day) > 0:
        #             break
        #         skipped = remaining.pop(0)
        #         print(f"  [WARN] HFR: no data from {skipped.name}, retrying with {len(remaining)} remaining...")

        #     if hfr_day is not None and len(hfr_day) > 0:
        #         hfr_dfs[sc][date_str] = hfr_day
        #         print(f"  Loaded HFR density ({sc}) {date_str}: {len(hfr_day)} samples, "
        #               f"Ne range [{hfr_day['Ne_cm3'].min():.2g}, {hfr_day['Ne_cm3'].max():.2g}] cm^-3")
        #     else:
        #         print(f"  [WARN] HFR density ({sc}) {date_str}: no usable data in any HFR file")
        # else:
        #     print(f"  HFR density ({sc}) {date_str}: no files found in {l2_folder}")
        
    
    current_date += timedelta(days=1)


###############################################################################
# SECTION 2: Echo detection
###############################################################################

for sc in ["RBSP-A", "RBSP-B"]:
    for date_str, items in waveform_data[sc].items():
        for item in items:
            t_rec, wf_bu, wf_bv, wf_bw, wf_eu, wf_ev, wf_ew, fs, event_utc, mage_day, wf_path = item
            
            for shape in range(wf_bu.shape[0]):
                B_wave = np.column_stack([
                    wf_bu[shape],
                    wf_bv[shape],
                    wf_bw[shape],
                    ])
                
                E_wave = np.column_stack([
                    wf_eu[shape],
                    wf_ev[shape],
                    wf_ew[shape],
                    ])
                
                # (1) Total magnetic/electric PSD
                freqs, spec_times, PB = total_field_psd(B_wave, fs)
                _, _, PE = total_field_psd(E_wave, fs)
                
                # (2) Mask to useful freq range
                freq_mask = (freqs >= 100) & (freqs <= 12000)
        
                freqs_use = freqs[freq_mask]
                PB_use = PB[:, freq_mask]
                PE_use = PE[:, freq_mask]
                
                # (3) Power threshold filter from paper x8
                PB_median = np.nanmedian(PB_use, axis=0)
                PE_median = np.nanmedian(PE_use, axis=0)
                
                power_mask = (
                    (PB_use >= 8.0 * PB_median[None, :]) &
                    (PE_use >= 8.0 * PE_median[None, :])
                    )
        
                # (4) Background magnetic field direction - can we get actual bg data?
                # b0_hat = B_wave / np.linalg.norm(B_wave)
                b0_hat = np.array([0.0, 0.0, 1.0])
                
                # (5) Get wna and ellipticity
                freqs_stft, spec_times_stft, wna, ellipticity = means_wna_ellipticity_from_waveform(
                    B_wave,
                    b0_hat=b0_hat,
                    fs=fs,
                    nperseg=1024,
                    noverlap=768,
                    nfft=1024,
                )
                
                # (6) Restrict WNA/ellipticity to same freq
                ellipticity_use = ellipticity[:, freq_mask]
                wna_use = wna[:, freq_mask]
                
                ##DEBUG
                mag_mask = PB_use >= 8.0 * PB_median[None, :]
                elec_mask = PE_use >= 8.0 * PE_median[None, :]
                ellip_mask = ellipticity_use >= 0.7
                
                print(
                    f"{sc} {event_utc} record {shape}: "
                    f"mag={np.sum(mag_mask)}, "
                    f"elec={np.sum(elec_mask)}, "
                    f"power={np.sum(mag_mask & elec_mask)}, "
                    f"ellip={np.sum(ellip_mask)}, "
                    f"final={np.sum(mag_mask & elec_mask & ellip_mask)}"
                )
                
                # lgw_mask = mag_mask & elec_mask & ellip_mask
                
                # (7) The paper mentions this
                lgw_mask = (
                    power_mask &
                    (ellipticity_use >= 0.7)
                )
                
                PB_filter = np.where(lgw_mask, PB_use, np.nan)
                WNA_filter = np.where(lgw_mask, wna_use, np.nan)
                ellipticity_filter = np.where(lgw_mask, ellipticity_use, np.nan)
                
                n_pass = int(np.sum(lgw_mask))
        
                print(
                    f"{sc} {event_utc} record {shape}: "
                    f"{n_pass} passing LGW pixels"
                )
                
                n_detected = int(np.sum(np.isfinite(PB_filter)))
                if n_detected:
                    plot_waveform_and_filtered(
                        times_records=t_rec,
                        waveform_2d=wf_bw,
                        fs=fs,
                        spacecraft=sc,
                        event_utc=event_utc,
                        record_number=shape + 1,
                        component="Bw",
                        duration_s=6.0,
                        fmax_hz=6000.0,
                        vmin=-13,
                        vmax=-7,
                        cmap="turbo",
                    
                        # Your detection outputs
                        filtered_psd=PB_filter,
                        filtered_freqs_hz=freqs_use,
                        filtered_times_s=spec_times,
                        filtered_label="Paper-style filtered LGW candidate pixels",
                    )

print("end")