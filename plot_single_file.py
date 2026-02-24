# -*- coding: utf-8 -*-
"""
Product-aware plotter for RBSP EMFISIS L2 CDFs.

Key fixes vs your previous version:
- Uses DEPEND_0 / DEPEND_1 metadata to find time and frequency axes (fixes *merged* products).
- Plots spectra + WFR diagonal products as spectrograms (time-frequency).
- Plots waveform products as RMS vs time + one example capture (with the L2 1kHz amp-only / no-phase warning).
- Plots magnetometer_uvw as U/V/W (+|B|) with downsampling for full-day viewing.

Run:
  python plot_single_file.py --base-dir RBSP-A/L2
Optional:
  python plot_single_file.py --base-dir RBSP-A/L2 --show-all-vars
  python plot_single_file.py --base-dir RBSP-A/L2 --save-dir out_plots
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cdflib


# ----------------------------
# Helpers
# ----------------------------
def list_cdf_files(day_dir: Path):
    return sorted(day_dir.glob("*.cdf"))


def list_variables(cdf: cdflib.CDF):
    info = cdf.cdf_info()
    vars_all = []
    for key in ("rVariables", "zVariables"):
        v = getattr(info, key, None)
        if v is None:
            continue
        if isinstance(v, dict):
            names = list(v.values())
        elif isinstance(v, (list, tuple)):
            names = list(v)
        else:
            try:
                names = list(v.values()) if hasattr(v, "values") else list(v)
            except Exception:
                names = []
        vars_all.extend(names)

    seen = set()
    out = []
    for n in vars_all:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def safe_str(x):
    """CDF attributes can come back as bytes, np.bytes_, lists, etc."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)


def var_data_type_is_tt2000(cdf: cdflib.CDF, varname: str) -> bool:
    """Prefer real TT2000 epoch detection over name-guessing."""
    try:
        vi = cdf.varinq(varname)
        # CDF_TIME_TT2000 is commonly 33 in CDF, but we avoid hard-coding only that:
        desc = str(vi.get("Data_Type_Description", "")).lower()
        dt = vi.get("Data_Type", None)
        return ("tt2000" in desc) or (dt == 33)
    except Exception:
        return False


def extract_tt2000_times(cdf: cdflib.CDF, time_var: str) -> pd.DatetimeIndex:
    t_raw = cdf.varget(time_var)
    times = cdflib.cdfepoch.to_datetime(t_raw)
    return pd.to_datetime(times)


def find_time_var(cdf: cdflib.CDF):
    """Robust TT2000 time discovery."""
    names = list_variables(cdf)

    # 1) direct TT2000 typed variables
    for n in names:
        if var_data_type_is_tt2000(cdf, n):
            try:
                _ = extract_tt2000_times(cdf, n)
                return n
            except Exception:
                pass

    # 2) fallback by common names
    for preferred in ("Epoch", "epoch", "Epoch_HFR", "Epoch_WFR", "Epoch_IWF"):
        if preferred in names:
            try:
                _ = extract_tt2000_times(cdf, preferred)
                return preferred
            except Exception:
                pass

    # 3) last resort heuristic
    candidates = [n for n in names if ("epoch" in n.lower() or "time" in n.lower())]
    for n in candidates:
        try:
            _ = extract_tt2000_times(cdf, n)
            return n
        except Exception:
            continue

    return None


def parse_product_from_filename(fname: str) -> str:
    # rbsp-a_<PRODUCT>_emfisis-L2_YYYYmmdd...cdf
    try:
        after_sc = fname.split("_", 1)[1]
        product = after_sc.split("_emfisis-", 1)[0]
        # Strip the optional hour tag in the date chunk e.g. 20121105T00 is not in product
        return product
    except Exception:
        return "UNKNOWN"


def downsample(x, max_points: int):
    x = np.asarray(x)
    n = x.shape[0]
    if n <= max_points:
        return x
    step = max(1, n // max_points)
    return x[::step]


def find_var_by_patterns(names, patterns):
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for n in names:
            if rx.search(n):
                return n
    return None


def get_depend_axes(cdf: cdflib.CDF, varname: str):
    """Return (depend0_time_var, depend1_freq_var) if present."""
    try:
        atts = cdf.varattsget(varname)
    except Exception:
        return (None, None)
    dep0 = safe_str(atts.get("DEPEND_0"))
    dep1 = safe_str(atts.get("DEPEND_1"))
    return dep0, dep1


def find_frequency_var(cdf: cdflib.CDF, data_nfreq: int):
    """Fallback frequency discovery if DEPEND_1 is missing."""
    names = list_variables(cdf)
    # Prefer variables with 'freq' in name and 1-D matching nfreq
    for n in names:
        if "freq" in n.lower():
            try:
                v = np.asarray(cdf.varget(n)).squeeze()
                if v.ndim == 1 and len(v) == data_nfreq:
                    return n
            except Exception:
                pass
    # Any 1-D matching length
    for n in names:
        try:
            v = np.asarray(cdf.varget(n)).squeeze()
            if v.ndim == 1 and len(v) == data_nfreq:
                return n
        except Exception:
            pass
    return None


# ----------------------------
# Plotters
# ----------------------------
def plot_spectrogram(times, freqs, data, title, freq_unit_hint=None, max_time_bins=6000):
    # data expected shape (Nt, Nf)
    times = pd.to_datetime(times)
    freqs = np.asarray(freqs).astype(float)

    # Downsample in time if huge
    Nt = data.shape[0]
    if Nt > max_time_bins:
        step = max(1, Nt // max_time_bins)
        data = data[::step, :]
        times = times[::step]

    # Avoid log(0)
    z = np.asarray(data).astype(float)
    z = np.maximum(z, np.nanmin(z[z > 0]) if np.any(z > 0) else 1e-30)
    z = np.log10(z)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    # pcolormesh wants edges-ish; shading='auto' usually fine
    ax.pcolormesh(times, freqs, z.T, shading="auto")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency" + (f" ({freq_unit_hint})" if freq_unit_hint else ""))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()


def plot_waveform_rms(times, wave_dict, title, note=None):
    """
    wave_dict: {varname: array (Nrecords, Nsamp)}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)

    # RMS per record
    times = pd.to_datetime(times)
    best_var = None
    best_idx = None
    best_rms = -np.inf

    for vname, arr in wave_dict.items():
        arr = np.asarray(arr)
        if arr.ndim < 2:
            continue
        rms = np.sqrt(np.mean(arr.astype(float) ** 2, axis=1))
        ax1.plot(times, rms, label=vname)

        i = int(np.nanargmax(rms))
        if rms[i] > best_rms:
            best_rms = float(rms[i])
            best_idx = i
            best_var = vname

    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("RMS (per capture)")
    ax1.legend(loc="upper right", fontsize=8)

    # Example capture (max RMS)
    if best_var is not None and best_idx is not None:
        ex = np.asarray(wave_dict[best_var])[best_idx, :].astype(float)
        ax2.plot(np.arange(len(ex)), ex)
        ax2.set_title(f"Example capture: {best_var} (record index {best_idx})")
        ax2.set_xlabel("Sample index")
        ax2.set_ylabel("Amplitude (L2 units)")

    if note:
        fig.text(0.01, 0.01, note, fontsize=9)

    plt.tight_layout()


def plot_magnetometer(times, vec, title, max_points=200000):
    times = pd.to_datetime(times)
    vec = np.asarray(vec).astype(float)

    # Downsample for sanity
    n = vec.shape[0]
    if n > max_points:
        step = max(1, n // max_points)
        vec = vec[::step, :]
        times = times[::step]

    Bu, Bv, Bw = vec[:, 0], vec[:, 1], vec[:, 2]
    Bmag = np.sqrt(Bu**2 + Bv**2 + Bw**2)

    plt.figure(figsize=(11, 4.5))
    plt.plot(times, Bu, label="B_u (UVW)")
    plt.plot(times, Bv, label="B_v (UVW)")
    plt.plot(times, Bw, label="B_w (UVW)")
    plt.plot(times, Bmag, label="|B|", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Magnetic field (L2 units)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()


def plot_housekeeping(times, y, yname, title, max_points=200000):
    times = pd.to_datetime(times)
    y = np.asarray(y).astype(float)

    if len(y) > max_points:
        step = max(1, len(y) // max_points)
        y = y[::step]
        times = times[::step]

    plt.figure(figsize=(11, 4.0))
    plt.plot(times, y)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(yname)
    plt.tight_layout()


# ----------------------------
# Core per-file handler
# ----------------------------
def inspect_all_variables(cdf: cdflib.CDF):
    print("All variables (name, shape, dtype, TT2000?):")
    for n in list_variables(cdf):
        try:
            vals = np.asarray(cdf.varget(n))
            tt = var_data_type_is_tt2000(cdf, n)
            print(f"  {n:40s} shape={str(vals.shape):16s} dtype={str(vals.dtype):12s} tt2000={tt}")
        except Exception:
            print(f"  {n:40s} (unreadable)")


def plot_cdf_file(fp: Path, show_all_vars: bool = False):
    cdf = cdflib.CDF(str(fp))
    names = list_variables(cdf)
    product = parse_product_from_filename(fp.name)

    print(f"  Product: {product}")

    if show_all_vars:
        inspect_all_variables(cdf)

    # -------------------------
    # HFR spectra (survey/burst/merged)
    # -------------------------
    if product.lower().startswith("hfr-spectra"):
        var = find_var_by_patterns(names, [r"^HFR_Spectra$", r"HFR.*Spectra"])
        if var is None:
            raise RuntimeError("Could not find an HFR spectra variable (expected something like HFR_Spectra).")

        dep0, dep1 = get_depend_axes(cdf, var)
        tvar = dep0 or find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for HFR spectra.")
        times = extract_tt2000_times(cdf, tvar)

        data = np.asarray(cdf.varget(var)).squeeze()
        if data.ndim != 2:
            raise RuntimeError(f"Expected 2-D spectra array for {var}, got shape {data.shape}")

        fvar = dep1 or find_frequency_var(cdf, data.shape[1])
        if not fvar:
            raise RuntimeError("Could not find frequency axis for HFR spectra.")
        freqs = np.asarray(cdf.varget(fvar)).squeeze()

        plot_spectrogram(times, freqs, data, title=fp.name, freq_unit_hint="Hz (10–400 kHz expected)")
        return

    # -------------------------
    # WFR spectral matrix diagonal (recommended)
    # -------------------------
    if product.lower().startswith("wfr-spectral-matrix-diagonal"):
        # Prefer BuBu (common), otherwise first 2D var with DEPEND_1
        var = find_var_by_patterns(names, [r"^BuBu$", r"^EuEu$", r"BuBu", r"EuEu"])
        if var is None:
            # find any 2-D var that has DEPEND_1
            for n in names:
                try:
                    v = np.asarray(cdf.varget(n)).squeeze()
                    if v.ndim == 2:
                        d0, d1 = get_depend_axes(cdf, n)
                        if d1:
                            var = n
                            break
                except Exception:
                    pass
        if var is None:
            raise RuntimeError("Could not identify a diagonal WFR spectral variable (e.g., BuBu).")

        dep0, dep1 = get_depend_axes(cdf, var)
        tvar = dep0 or find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for WFR spectral matrix diagonal.")
        times = extract_tt2000_times(cdf, tvar)

        data = np.asarray(cdf.varget(var)).squeeze()
        if data.ndim != 2:
            raise RuntimeError(f"Expected 2-D array for {var}, got shape {data.shape}")

        fvar = dep1 or find_frequency_var(cdf, data.shape[1])
        if not fvar:
            raise RuntimeError("Could not find frequency axis for WFR diagonal spectral matrix.")
        freqs = np.asarray(cdf.varget(fvar)).squeeze()

        plot_spectrogram(times, freqs, data, title=f"{fp.name} (showing {var})", freq_unit_hint="Hz (2–12 kHz expected)")
        return

    # -------------------------
    # WFR waveform products (survey/burst/continuous-burst)
    # -------------------------
    if product.lower().startswith("wfr-waveform"):
        # Grab all *Samples vars that are 2-D (records x samples)
        wave_vars = []
        for n in names:
            if n.lower().endswith("samples"):
                try:
                    arr = np.asarray(cdf.varget(n))
                    if arr.ndim >= 2:
                        wave_vars.append(n)
                except Exception:
                    pass

        if not wave_vars:
            raise RuntimeError("No waveform *Samples variables found for WFR waveform product.")

        # Time axis: use DEPEND_0 from the first waveform channel, else TT2000 discovery
        dep0, _ = get_depend_axes(cdf, wave_vars[0])
        tvar = dep0 or find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for WFR waveform.")
        times = extract_tt2000_times(cdf, tvar)

        wave_dict = {}
        for n in wave_vars[:6]:  # keep it readable; still includes multiple axes
            arr = np.asarray(cdf.varget(n)).squeeze()
            if arr.ndim == 2 and arr.shape[0] == len(times):
                wave_dict[n] = arr

        note = ("WARNING: L2 waveform data is amplitude-calibrated at 1 kHz only; "
                "no phase calibration is applied. For wave-parameter work, apply the "
                "complex frequency adjustment table after FFT (per EMFISIS docs).")
        plot_waveform_rms(times, wave_dict, title=fp.name, note=note)
        return

    # -------------------------
    # HFR waveform
    # -------------------------
    if product.lower().startswith("hfr-waveform"):
        var = find_var_by_patterns(names, [r"HFRsamples", r"HFR.*samples"])
        if var is None:
            raise RuntimeError("Could not find HFR waveform samples variable (expected HFRsamples).")

        dep0, _ = get_depend_axes(cdf, var)
        tvar = dep0 or find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for HFR waveform.")
        times = extract_tt2000_times(cdf, tvar)

        arr = np.asarray(cdf.varget(var)).squeeze()
        if arr.ndim == 1:
            # simple 1D series
            plot_housekeeping(times, arr, var, fp.name)
            return
        if arr.ndim == 2 and arr.shape[0] == len(times):
            # Treat like waveform captures
            wave_dict = {var: arr}
            plot_waveform_rms(times, wave_dict, title=fp.name)
            return

        raise RuntimeError(f"Unexpected HFR waveform shape for {var}: {arr.shape}")

    # -------------------------
    # Magnetometer UVW
    # -------------------------
    if product.lower().startswith("magnetometer_uvw"):
        # Prefer a 2D vector variable (N,3). If not found, fall back.
        tvar = find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for magnetometer.")
        times = extract_tt2000_times(cdf, tvar)

        vec_var = None
        vec = None

        # Try common vector storage: one var with 3 columns
        for n in names:
            try:
                v = np.asarray(cdf.varget(n)).squeeze()
                if v.ndim == 2 and v.shape[0] == len(times) and v.shape[1] == 3:
                    if "mag" in n.lower() or "b" in n.lower():
                        vec_var = n
                        vec = v
                        break
            except Exception:
                pass

        # If not found, try assembling from separate components
        if vec is None:
            bu = find_var_by_patterns(names, [r"^Bu$", r"Bu$", r"^B_u$", r"b_u"])
            bv = find_var_by_patterns(names, [r"^Bv$", r"Bv$", r"^B_v$", r"b_v"])
            bw = find_var_by_patterns(names, [r"^Bw$", r"Bw$", r"^B_w$", r"b_w"])
            if bu and bv and bw:
                Bu = np.asarray(cdf.varget(bu)).squeeze()
                Bv = np.asarray(cdf.varget(bv)).squeeze()
                Bw = np.asarray(cdf.varget(bw)).squeeze()
                vec = np.vstack([Bu, Bv, Bw]).T
                vec_var = f"{bu},{bv},{bw}"

        if vec is None:
            # fallback: plot any 1D mag-like variable
            y = None
            yname = None
            for n in names:
                if "mag" in n.lower():
                    try:
                        v = np.asarray(cdf.varget(n)).squeeze()
                        if v.ndim == 1 and len(v) == len(times):
                            y, yname = v, n
                            break
                    except Exception:
                        pass
            if y is None:
                raise RuntimeError("Could not identify magnetometer vector or scalar series.")
            plot_housekeeping(times, y, yname, fp.name)
            return

        plot_magnetometer(times, vec, title=f"{fp.name} (UVW spinning frame)", max_points=250000)
        return

    # -------------------------
    # Housekeeping (health/status)
    # -------------------------
    if product.lower().startswith("housekeeping"):
        tvar = find_time_var(cdf)
        if not tvar:
            raise RuntimeError("No TT2000 time variable found for housekeeping.")
        times = extract_tt2000_times(cdf, tvar)

        # Prefer the +5.5V current monitor if present, else first numeric 1D aligned var
        preferred = find_var_by_patterns(names, [r"\+5\.5V.*Current", r"Current Mon", r"Voltage"])
        candidates = [preferred] if preferred else []

        if not candidates:
            for n in names:
                if n == tvar:
                    continue
                try:
                    v = np.asarray(cdf.varget(n)).squeeze()
                    if v.ndim == 1 and len(v) == len(times) and np.issubdtype(v.dtype, np.number):
                        candidates.append(n)
                        break
                except Exception:
                    pass

        if not candidates:
            raise RuntimeError("No numeric housekeeping variable aligned to time found.")

        yname = candidates[0]
        y = np.asarray(cdf.varget(yname)).squeeze()
        plot_housekeeping(times, y, yname, fp.name)
        return

    # -------------------------
    # Fallback (unknown product)
    # -------------------------
    # Try to find a TT2000 time + a numeric aligned 1D variable to plot as a last resort.
    tvar = find_time_var(cdf)
    if not tvar:
        raise RuntimeError("No TT2000 time variable found (fallback).")
    times = extract_tt2000_times(cdf, tvar)

    yname, y = None, None
    for n in names:
        if n == tvar:
            continue
        try:
            v = np.asarray(cdf.varget(n)).squeeze()
            if v.ndim == 1 and len(v) == len(times) and np.issubdtype(v.dtype, np.number):
                yname, y = n, v
                break
        except Exception:
            pass

    if y is None:
        raise RuntimeError("Fallback could not find any aligned numeric 1D variable to plot.")

    plot_housekeeping(times, y, yname, fp.name)


# ----------------------------
# CLI
# ----------------------------
def choose_date():
    default_date = "2012-11-05"
    user_input = input(f"Enter date (YYYY-MM-DD) or press Enter for default [{default_date}]: ").strip()
    if not user_input:
        user_input = default_date
    try:
        date_obj = pd.to_datetime(user_input)
        return date_obj.strftime("%Y/%m/%d")
    except Exception as e:
        print(f"Invalid date format: {e}. Using default {default_date}")
        return pd.to_datetime(default_date).strftime("%Y/%m/%d")


def main():
    parser = argparse.ArgumentParser(description="Plot all EMFISIS CDFs in a single day folder (product-aware).")
    parser.add_argument("--base-dir", default="RBSP-B/L2", help="Base directory (e.g., RBSP-A/L2)")
    parser.add_argument("--show-all-vars", action="store_true", help="Print all variable names + shapes per file")
    parser.add_argument("--save-dir", default=None, help="If set, save each figure as PNG into this folder")
    args = parser.parse_args()

    date_str = choose_date()
    day_dir = Path(args.base_dir) / date_str

    print(f"\n{'='*60}")
    print(f"Opening folder: {day_dir}")
    print(f"{'='*60}")

    if not day_dir.is_dir():
        print(f"Directory not found: {day_dir}")
        return

    files = list_cdf_files(day_dir)
    if not files:
        print(f"No CDF files in {day_dir}")
        return

    print(f"Number of CDF files: {len(files)}\n")

    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = None

    for i, fp in enumerate(files, 1):
        print(f"\n{'-'*60}")
        print(f"File {i}/{len(files)}: {fp.name}")
        print(f"File size: {fp.stat().st_size / 1024:.2f} KB")

        try:
            plot_cdf_file(fp, show_all_vars=args.show_all_vars)

            if outdir is not None:
                safe_name = fp.name.replace(".cdf", ".png")
                plt.savefig(outdir / safe_name, dpi=150)
                plt.close("all")

        except Exception as e:
            print(f"Error processing {fp.name}: {e}")

    if outdir is None:
        plt.show()

    print("done")


if __name__ == "__main__":            
    main()
