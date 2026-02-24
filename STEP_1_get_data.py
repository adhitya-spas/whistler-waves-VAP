# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, Sequence, List
import subprocess
import pandas as pd
import numpy as np
import h5py
from cdflib import CDF, cdfepoch

# ====================================================================================================
# Main Functions
# ====================================================================================================

def check_available_files(
    spacecraft: str,
    start_time: str,
    end_time: str,
    root_dir: str = r"F:\UAF_Research",
    magephem_model: str = "T89D",
) -> Dict[str, Dict]:
    
    results = {
        "emfisis_spectral": {},
        "emfisis_waveform": {},
        "magephem": {},
    }
    
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    print(f"Checking for EMFSIS L2 files for {spacecraft} from {start_time} to {end_time}...")
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        date_str = f"{current_date:%Y-%m-%d}"
        folder = Path(root_dir) / spacecraft / "L2" / f"{current_date:%Y}" / f"{current_date:%m}" / f"{current_date:%d}"
        
        files = []
        if folder.is_dir():
            files = list(folder.glob("*.cdf"))
            if files:
                print(f"Found {len(files)} files in {folder}")
        
        if not files:
            batch_path = Path(root_dir) / "get_emfsis_one_day.bat"
            subprocess.run(["cmd.exe", "/c", str(batch_path), spacecraft, date_str], check=False)
            if folder.is_dir():
                files = list(folder.glob("*.cdf"))
        
        spectral_files = [p for p in files if "spectral-matrix" in p.name]
        waveform_files = [p for p in files if "WFR-waveform-continuous-burst" in p.name]
        
        results["emfisis_spectral"][date_str] = spectral_files
        results["emfisis_waveform"][date_str] = waveform_files
        current_date += timedelta(days=1)

    print(f"Checking for MagEphem files for {spacecraft} from {start_time} to {end_time}...")
    sc_prefix = spacecraft.lower().replace("-", "")
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        date_str = f"{current_date:%Y-%m-%d}"
        folder = Path(root_dir) / spacecraft / "MagEphem" / f"{current_date:%Y}" / f"{current_date:%m}" / f"{current_date:%d}"
        
        h5_candidates = []
        if folder.is_dir():
            h5_candidates = [f for f in folder.glob("*.h5") if sc_prefix in f.name.lower() and magephem_model in f.name]
            if h5_candidates:
                print(f"Found files in {folder}")
        
        if not h5_candidates:
            batch_path = Path(root_dir) / "get_magephem_one_day.bat"
            subprocess.run(["cmd.exe", "/c", str(batch_path), spacecraft, date_str], check=False)
            if folder.is_dir():
                h5_candidates = [f for f in folder.glob("*.h5") if sc_prefix in f.name.lower() and magephem_model in f.name]
        
        results["magephem"][date_str] = {
            "h5": max(h5_candidates, key=lambda p: tuple(map(int, p.stem.split("_v")[-1].split(".")))) if h5_candidates else None
        }
        current_date += timedelta(days=1)
    
    return results

def read_magephem_h5(h5_path: str | Path) -> pd.DataFrame:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        cols = _flatten_h5_group_to_columns(f)

    lengths = [len(v) for v in cols.values() 
               if isinstance(v, (list, tuple, np.ndarray, pd.Series)) 
               and not isinstance(v, (str, bytes, bytearray))]
    
    if not lengths:
        return pd.DataFrame()

    time_cols, _ = split_time_vs_meta(cols, time_len=max(lengths))
    return pd.DataFrame(time_cols)

def read_emfisis_spectral_merged_files(
    cdf_paths: Sequence[Path],
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:

    def _safe_str(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        if isinstance(x, (bytes, np.bytes_)):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return str(x)
        return str(x)

    def _last_rec(cdf, v):
        return getattr(cdf.varinq(v), "Last_Rec", -1)

    cdf_paths = [Path(p) for p in cdf_paths if p is not None and Path(p).exists()]
    if not cdf_paths:
        raise FileNotFoundError("No existing spectral CDF files provided.")

    all_times, all_data = [], []
    freq_ref = None

    for fp in sorted(cdf_paths):
        with CDF(str(fp)) as cdf:
            info = cdf.cdf_info()
            names = list(getattr(info, "zVariables", []) or []) + list(getattr(info, "rVariables", []) or [])

            spec_var = (
                    "BwBw" if "BwBw" in names else
                    ("BuBu" if "BuBu" in names else
                     ("EuEu" if "EuEu" in names else None))
                )
            if not spec_var:
                continue

            if _last_rec(cdf, spec_var) < 0:
                continue
            atts = cdf.varattsget(spec_var)
            tvar = _safe_str(atts.get("DEPEND_0")) or ("Epoch" if "Epoch" in names else "")
            if not tvar or tvar not in names or _last_rec(cdf, tvar) < 0:
                continue

            fvar = _safe_str(atts.get("DEPEND_1"))
            if not fvar or fvar not in names:
                continue

            times = pd.to_datetime(cdfepoch.to_datetime(cdf.varget(tvar)), utc=True)
            freqs = np.asarray(cdf.varget(fvar)).squeeze()
            data  = np.asarray(cdf.varget(spec_var)).squeeze()

            if data.ndim != 2 or data.shape[0] != len(times) or data.shape[1] != len(freqs):
                continue
            pos = data[data > 0]
            floor = np.nanmin(pos) if pos.size else 1e-30
            data_log = np.log10(np.maximum(data, floor))

            all_times.append(times)
            all_data.append(data_log)
            if freq_ref is None:
                freq_ref = freqs

    if not all_times:
        raise RuntimeError("No usable spectral data found in provided files.")

    times_out = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in all_times]), tz="UTC")
    data_stacked = np.vstack(all_data)

    order = np.argsort(times_out.view("i8"))
    return times_out[order], freq_ref, data_stacked[order, :]

def read_wfr_waveform_continuous_burst(
    cdf_path: Path,
    component: str = "Bw",
) -> Tuple[pd.DatetimeIndex, np.ndarray, float]:

    def _safe_str(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        if isinstance(x, (bytes, np.bytes_)):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return str(x)
        return str(x)

    def _is_tt2000(cdf: CDF, varname: str) -> bool:
        try:
            info = cdf.varinq(varname)
            return getattr(info, "Data_Type", None) == 33
        except Exception:
            return False

    cdf_path = Path(cdf_path)
    with CDF(str(cdf_path)) as cdf:
        info = cdf.cdf_info()
        names = list(getattr(info, "zVariables", []) or []) + list(getattr(info, "rVariables", []) or [])

        comp_lower = component.lower()
        candidates = [n for n in names if comp_lower in n.lower() and "sample" in n.lower()]
        if not candidates:
            candidates = [n for n in names if comp_lower in n.lower()]
        if not candidates:
            raise RuntimeError(f"No waveform component found matching '{component}' in {cdf_path.name}")
        wvar = candidates[0]

        atts = cdf.varattsget(wvar)
        tvar = _safe_str(atts.get("DEPEND_0")) if atts else ""
        if not tvar or tvar not in names or not _is_tt2000(cdf, tvar):
            if "Epoch" in names and _is_tt2000(cdf, "Epoch"):
                tvar = "Epoch"
            else:
                tt_candidates = [n for n in names if _is_tt2000(cdf, n)]
                tvar = tt_candidates[0] if tt_candidates else None
        if not tvar:
            raise RuntimeError(f"No valid time variable found in {cdf_path.name}")

        times = pd.to_datetime(cdfepoch.to_datetime(cdf.varget(tvar)), utc=True)
        data = np.asarray(cdf.varget(wvar))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

    fs = 35000.0
    return pd.DatetimeIndex(times, tz="UTC"), data, fs

def read_emfisis_hfr_density_files(
    hfr_files: list,
    skip_low_bins: int = 3,
    skip_high_bins: int = 3,
    noise_floor: float = 1e-20,
    median_window: int = 61,
    f_ce_hz_series: pd.Series | np.ndarray | None = None,
    min_fuh_hz: float = 10e3,
    use_hysteresis: bool = True,
    hysteresis_halfwidth: int = 7,
    disable_low_f_clip: bool = False,
) -> pd.DataFrame:
    _TIME_CANDIDATES = ["Epoch", "epoch", "UTC", "Time", "time"]
    _SPEC_CANDIDATES = ["HFR_Spectra", "hfr_spectra", "Spectra", "spectra"]
    _FREQ_CANDIDATES = ["HFR_frequencies", "hfr_frequencies",
                        "HFR_Frequency", "Frequency", "frequencies", "freqs"]
    
    # Kurth et al. 2015 constant: f_pe [Hz] = 8980 * sqrt(Ne [cm^-3])
    FPE_CONVERSION_CONST = 8980.0  # Hz·cm^(-3/2)

    frames = []
    for fp in hfr_files:
        fp = Path(fp)
        if not fp.exists():
            print(f"[WARN] HFR: file not found, skipping: {fp}")
            continue
        try:
            with CDF(str(fp)) as cdf:
                info  = cdf.cdf_info()
                names = set(
                    list(getattr(info, "zVariables", []) or []) +
                    list(getattr(info, "rVariables", []) or [])
                )

                tvar = next((v for v in _TIME_CANDIDATES if v in names), None)
                if tvar is None:
                    print(f"[WARN] HFR: no time variable in {fp.name}, skipping")
                    continue
                raw_t = cdf.varget(tvar)
                times = pd.to_datetime(cdfepoch.to_datetime(raw_t), utc=True)

                spec_var = next((v for v in _SPEC_CANDIDATES if v in names), None)
                freq_var = next((v for v in _FREQ_CANDIDATES if v in names), None)
                if spec_var is None or freq_var is None:
                    print(f"[WARN] HFR: spectra/frequency variable not found in "
                          f"{fp.name} (vars: {sorted(names)}), skipping")
                    continue

                spectra = np.asarray(cdf.varget(spec_var), dtype=float)
                freqs_hz = np.asarray(cdf.varget(freq_var), dtype=float).squeeze()

                if spectra.ndim == 1:
                    spectra = spectra.reshape(1, -1)
                elif spectra.shape[0] == len(freqs_hz) and spectra.shape[1] != len(freqs_hz):
                    spectra = spectra.T

                assert spectra.ndim == 2, f"Expected 2-D spectra, got {spectra.ndim}D"
                assert spectra.shape[1] == len(freqs_hz), \
                    f"Spectra shape {spectra.shape} incompatible with freqs length {len(freqs_hz)}"

                n_freqs = spectra.shape[1]

                i_lo = skip_low_bins
                i_hi = n_freqs - skip_high_bins if skip_high_bins > 0 else n_freqs
                if i_lo >= i_hi:
                    i_lo, i_hi = 0, n_freqs  # fallback: use all bins

                spec_trim = spectra[:, i_lo:i_hi].copy()
                freqs_trim = freqs_hz[i_lo:i_hi].copy()

                freq_max = np.nanmax(freqs_trim)
                freq_min = np.nanmin(freqs_trim)
                freq_median = np.nanmedian(freqs_trim)
                
                if freq_max < 5000:
                    freqs_trim = freqs_trim * 1e3
                
                assert spec_trim.shape == (len(times), len(freqs_trim)), \
                    f"spec_trim shape mismatch: {spec_trim.shape} vs (Nt={len(times)}, Nf={len(freqs_trim)})"
                assert freqs_trim.ndim == 1, f"freqs_trim must be 1-D, got {freqs_trim.ndim}D"

                if noise_floor > 0:
                    spec_trim[spec_trim < noise_floor] = np.nan

                if use_hysteresis:
                    peak_idx = _peak_track_with_hysteresis(
                        spec_trim, freqs_trim, search_halfwidth=hysteresis_halfwidth
                    )
                else:
                    peak_idx = np.full(len(times), np.nan)
                    for i in range(len(times)):
                        row = spec_trim[i, :]
                        if np.any(np.isfinite(row)):
                            peak_idx[i] = np.nanargmax(row)

                valid = np.isfinite(peak_idx)
                f_uh_hz = np.full(len(times), np.nan)
                f_uh_hz[valid] = freqs_trim[peak_idx[valid].astype(int)]

                f_pe_hz = np.full(len(times), np.nan)
                ne_cm3 = np.full(len(times), np.nan)
                quality_flag = np.full(len(times), 0, dtype=int)
                used_fce_correction = np.zeros(len(times), dtype=bool)

                f_ce_hz_at_times = None
                if f_ce_hz_series is not None:
                    if isinstance(f_ce_hz_series, pd.Series):
                        f_ce_hz_at_times = f_ce_hz_series.reindex(times, method='nearest')
                        f_ce_hz_at_times = f_ce_hz_at_times.to_numpy()
                    elif isinstance(f_ce_hz_series, np.ndarray):
                        if len(f_ce_hz_series) == len(times):
                            f_ce_hz_at_times = f_ce_hz_series.copy()
                        else:
                            print(f"  [WARN] f_ce_hz_series length {len(f_ce_hz_series)} "
                                  f"!= HFR times length {len(times)}, ignoring f_ce")

                for i in range(len(times)):
                    if not np.isfinite(f_uh_hz[i]):
                        quality_flag[i] = 3
                        continue
                    
                    if not disable_low_f_clip and f_uh_hz[i] < min_fuh_hz:
                        quality_flag[i] = 1
                        f_uh_hz[i] = np.nan
                        continue
                    
                    if f_ce_hz_at_times is not None and np.isfinite(f_ce_hz_at_times[i]):
                        f_ce = f_ce_hz_at_times[i]
                        
                        if f_uh_hz[i] <= f_ce:
                            quality_flag[i] = 2
                            f_uh_hz[i] = np.nan
                            continue
                        
                        f_pe_hz[i] = np.sqrt(max(f_uh_hz[i]**2 - f_ce**2, 0.0))
                        used_fce_correction[i] = True
                    else:
                        f_pe_hz[i] = f_uh_hz[i]
                        used_fce_correction[i] = False
                    
                    if np.isfinite(f_pe_hz[i]):
                        ne_cm3[i] = (f_pe_hz[i] / FPE_CONVERSION_CONST) ** 2

                if median_window > 1 and np.any(np.isfinite(ne_cm3)):
                    from scipy.ndimage import median_filter
                    w = median_window if median_window % 2 == 1 else median_window + 1
                    ne_smooth = median_filter(np.where(np.isfinite(ne_cm3), ne_cm3, np.nan), size=w)
                    ne_smooth[~np.isfinite(ne_cm3)] = np.nan
                    ne_cm3 = ne_smooth

                df_file = pd.DataFrame(
                    {
                        "f_uh_hz": f_uh_hz,
                        "f_pe_hz": f_pe_hz,
                        "Ne_cm3": ne_cm3,
                        "quality_flag": quality_flag,
                        "used_fce_correction": used_fce_correction,
                    },
                    index=times
                )
                df_file = df_file[df_file["Ne_cm3"] > 0]
                frames.append(df_file)

        except Exception as exc:
            print(f"[WARN] HFR: failed to read {fp.name}: {exc}")
            import traceback; traceback.print_exc()
            continue

    if not frames:
        return pd.DataFrame(
            columns=["f_uh_hz", "f_pe_hz", "Ne_cm3", "quality_flag", "used_fce_correction"]
        )

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    
    return out

# ====================================================================================================
# Support Functions
# ====================================================================================================

def find_emfisis_hfr_cdf_files(
    spacecraft: str,
    start_time: str,
    end_time: str,
    root_dir: str = r"F:\UAF_Research",
) -> List[Path]:
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    found: List[Path] = []
    current_date = start_dt.date()

    while current_date <= end_dt.date():
        date_str = f"{current_date:%Y-%m-%d}"
        folder   = (
            Path(root_dir) / spacecraft / "L2"
            / f"{current_date:%Y}" / f"{current_date:%m}" / f"{current_date:%d}"
        )

        def _hfr_cdfs(directory: Path) -> List[Path]:
            if not directory.is_dir():
                return []
            return [
                p for p in directory.glob("*.cdf")
                if "hfr" in p.name.lower() and "spectra" in p.name.lower()
            ]

        day_files = _hfr_cdfs(folder)

        if not day_files:
            print(f"  [HFR] No local files for {date_str}, attempting download…")
            batch_path = Path(root_dir) / "get_emfsis_one_day.bat"
            subprocess.run(
                ["cmd.exe", "/c", str(batch_path), spacecraft, date_str],
                check=False,
            )
            day_files = _hfr_cdfs(folder)

        found.extend(day_files)
        current_date += timedelta(days=1)

    found_sorted = sorted(set(found))
    print(f"[HFR] {spacecraft} {start_time[:10]}→{end_time[:10]}: {len(found_sorted)} HFR file(s) found.")
    return found_sorted

def get_hfr_density_df(
    spacecraft: str,
    start_time: str,
    end_time: str,
    root_dir: str = r"F:\UAF_Research",
) -> pd.DataFrame:

    hfr_files = find_emfisis_hfr_cdf_files(
        spacecraft=spacecraft,
        start_time=start_time,
        end_time=end_time,
        root_dir=root_dir,
    )

    if not hfr_files:
        print(f"[HFR] No HFR files available — returning empty density DataFrame.")
        return pd.DataFrame(columns=["Ne_cm3"])

    df = read_emfisis_hfr_density_files(
        hfr_files=hfr_files,
        skip_low_bins=3,
        skip_high_bins=3,
        noise_floor=1e-22,
        median_window=61,
    )

    if df.empty:
        print("[HFR] Density derivation returned no valid rows.")
    else:
        print(f"[HFR] Density DataFrame: {len(df)} rows, "
              f"Ne_cm3 median={df['Ne_cm3'].median():.2f} cm⁻³, "
              f"range=[{df['Ne_cm3'].min():.2f}, {df['Ne_cm3'].max():.2f}]")
    return df

def _flatten_h5_group_to_columns(h5f) -> Dict:
    cols = {}

    def to_str(x):
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            try:
                return bytes(x).decode("utf-8", errors="replace")
            except:
                return str(x)
        return str(x)

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            data = obj[()]
        except:
            return

        if isinstance(data, list):
            data = np.asarray(data, dtype=object)

        if isinstance(data, (bytes, bytearray, np.bytes_)):
            cols[name] = to_str(data)
            return
        if isinstance(data, np.generic):
            item = data.item()
            cols[name] = to_str(item) if isinstance(item, (bytes, bytearray, np.bytes_)) else item
            return

        arr = np.asarray(data)

        if arr.dtype.kind in ("S", "U"):
            flat = arr.reshape(-1)
            cols[name] = np.asarray([to_str(x) for x in flat], dtype=object)
            return

        if arr.dtype.kind == "O":
            flat = arr.reshape(-1)
            has_bytes = any(isinstance(x, (bytes, bytearray, np.bytes_)) for x in flat[:min(len(flat), 50)])
            if has_bytes:
                cols[name] = np.asarray([to_str(x) for x in flat], dtype=object)
            else:
                cols[name] = arr.reshape(-1)
            return

        if arr.ndim == 2:
            for i in range(arr.shape[1]):
                cols[f"{name}_{i}"] = arr[:, i]
        elif arr.ndim == 1:
            cols[name] = arr
        else:
            cols[name] = arr.reshape(-1)

    h5f.visititems(visit)
    return cols

def _peak_track_with_hysteresis(
    spec_trim: np.ndarray,
    freqs_trim: np.ndarray,
    search_halfwidth: int = 7,
) -> np.ndarray:
    peak_idx = np.full(len(spec_trim), np.nan)
    
    for i in range(len(spec_trim)):
        row = spec_trim[i, :]
        
        if not np.any(np.isfinite(row)):
            continue
        
        if i == 0 or not np.isfinite(peak_idx[i - 1]):
            peak_idx[i] = np.nanargmax(row)
        else:
            prev_peak = int(peak_idx[i - 1])
            search_lo = max(0, prev_peak - search_halfwidth)
            search_hi = min(len(row), prev_peak + search_halfwidth + 1)
            
            search_region = row[search_lo:search_hi]
            if np.any(np.isfinite(search_region)):
                local_max_idx = np.nanargmax(search_region)
                peak_idx[i] = search_lo + local_max_idx
            else:
                peak_idx[i] = np.nanargmax(row)
    
    return peak_idx

def split_time_vs_meta(cols: dict, time_len: int):
    time_cols = {}
    meta = {}
    for k, v in cols.items():
        if not hasattr(v, "__len__") or isinstance(v, (str, bytes, bytearray)):
            meta[k] = v
            continue
        L = len(v)
        if L == time_len:
            time_cols[k] = v
        else:
            meta[k] = v
    return time_cols, meta