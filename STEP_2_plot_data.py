# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle, Wedge
from matplotlib.patheffects import Stroke, Normal
import scipy.signal as signal
from datetime import timedelta
from STEP_1_get_data import get_hfr_density_df

# Constants
RE_KM = 6371.0
IONO_ALT_KM = 90.0
_FCE_KHZ_PER_NT = 0.02799248987233304
_FPE_KHZ_CONVERSION = 8.98

# ====================================================================================================
# Main Functions
# ====================================================================================================

def plot_emfsis_waveform(
    times_records,
    waveform_2d,
    fs: float,
    spacecraft: str,
    event_utc: str | None = None,
    record_number: int | None = None,
    magephem_df=None,
    component: str = "Bw",
    duration_s=6.0,
    fmax_hz=6000.0,
    vmin=-13,
    vmax=-7,
    cmap="turbo",
    figsize=(9.6, 4.6),
    show_fce: bool = False,
    outpath: str | None = None,
):
    t_rec = pd.to_datetime(times_records, utc=True)
    if len(t_rec) == 0:
        print("[WARN] No waveform records available")
        return
    
    if record_number is not None:
        idx = max(0, min(int(record_number) - 1, len(t_rec) - 1))
        event_dt = t_rec[idx]
    else:
        event_dt = pd.to_datetime(event_utc, utc=True) if event_utc is not None else t_rec[0]
        idx = int(np.argmin(np.abs((t_rec - event_dt).to_numpy(dtype="timedelta64[ns]"))))
    
    w = np.asarray(waveform_2d)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if idx >= w.shape[0]:
        idx = w.shape[0] - 1
    w_sel = np.asarray(w[idx], dtype=float).flatten()
    
    f_hz, t_sec, psd = signal.spectrogram(
        w_sel,
        fs=fs,
        nperseg=1024,
        noverlap=1000,
        nfft=1024,
        scaling="density",
        mode="psd",
    )
    
    t_mask = (t_sec >= 0) & (t_sec <= duration_s)
    f_mask = (f_hz >= 0) & (f_hz <= fmax_hz)
    if not np.any(t_mask) or not np.any(f_mask):
        print("[WARN] Spectrogram window is empty")
        return
    
    t_plot = t_sec[t_mask]
    f_plot = f_hz[f_mask] / 1e3
    Z = psd[np.ix_(f_mask, t_mask)]
    Z = np.maximum(Z, 1e-30)
    Z_log = np.log10(Z)
    
    header = f"{spacecraft}  {event_dt.strftime('%d %b %Y %H:%M:%S')} UT  (rec {idx+1}/{w.shape[0]})"
    
    mage_meta = {}
    if magephem_df is not None and len(magephem_df) > 0:
        mage_meta = sample_magephem_at_time(magephem_df, event_dt)
    
    meta_parts = []
    if mage_meta.get("Lsimple"):
        meta_parts.append(f"L={mage_meta['Lsimple']:.2f}")
    if mage_meta.get("CDMAG_MLAT") is not None:
        meta_parts.append(f"λm={mage_meta['CDMAG_MLAT']:.1f}°")
    if mage_meta.get("CDMAG_MLT") is not None:
        meta_parts.append(f"MLT={mage_meta['CDMAG_MLT']:.1f}h")
    if mage_meta.get("InOut") is not None:
        meta_parts.append(f"InOut={mage_meta['InOut']}")
    if mage_meta.get("Bmag") is not None:
        meta_parts.append(f"|B|={mage_meta['Bmag']:.1f}nT")
    if mage_meta.get("BoverBeq") is not None:
        meta_parts.append(f"B/Beq={mage_meta['BoverBeq']:.2f}")
    if mage_meta.get("Kp") is not None:
        meta_parts.append(f"Kp={mage_meta['Kp']:.1f}")
    if mage_meta.get("Dst") is not None:
        meta_parts.append(f"Dst={mage_meta['Dst']:.0f}nT")
    
    metadata_line = "  ".join(meta_parts) if meta_parts else "(no MagEphem data)"
    
    fig, ax_spec = plt.subplots(figsize=figsize, constrained_layout=False)
    
    # Plot spectrogram
    im = ax_spec.pcolormesh(t_plot, f_plot, Z_log, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax_spec.set_xlim(0, duration_s)
    ax_spec.set_ylim(0.01, fmax_hz / 1e3)
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (kHz)")
    
    fig.suptitle("EMFISIS Waveform Spectrogram", fontsize=14, fontweight="bold", y=0.98)
    
    title_with_meta = f"{header}\n{metadata_line}"
    ax_spec.set_title(title_with_meta, fontsize=10, pad=10)
    
    cbar = plt.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.046)
    ticks = [-13, -11, -9, -7]
    ticks = [tv for tv in ticks if vmin <= tv <= vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{int(tv)}}}$" for tv in ticks])
    cbar.set_label(rf"{component} PSD (nT²/Hz)")
    
    plt.tight_layout()
    
    plt.show()

def plot_magephem_day_context(
    spacecraft: str,
    date_str: str,
    mage_df,
    t_start,
    t_end,
    burst_times=None,
    out_dir: str | None = None,
):
    """
    Create a single day-long MagEphem context plot covering the spectral time range.
    
    Uses 3 stacked subplots:
    1) Lsimple + CDMAG_MLAT (twin y-axis)
    2) CDMAG_MLT
    3) |B| + BoverBeq (twin y-axis)
    
    Saves figure if out_dir is provided.
    """
    if mage_df is None or len(mage_df) == 0:
        print("[WARN] No MagEphem data for day context plot")
        return
    
    df = _magephem_index(mage_df)
    if df is None or df.empty:
        print("[WARN] MagEphem index failed for day context plot")
        return
    
    t0 = pd.to_datetime(t_start, utc=True, errors="coerce")
    t1 = pd.to_datetime(t_end, utc=True, errors="coerce")
    if pd.isna(t0) or pd.isna(t1):
        print("[WARN] Invalid time range for day context plot")
        return
    
    win = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
    if win.empty:
        print("[WARN] No MagEphem samples in day context window")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=False)
    ax1, ax2, ax3 = axes
    
    # Panel 1: Lsimple + CDMAG_MLAT
    if "Lsimple" in win.columns:
        L = pd.to_numeric(win["Lsimple"], errors="coerce")
        ax1.plot(win.index, L, label="Lsimple", color="tab:blue", linewidth=1.2)
        ax1.set_ylabel("Lsimple", fontsize=11, color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    if "CDMAG_MLAT" in win.columns:
        mlat = pd.to_numeric(win["CDMAG_MLAT"], errors="coerce")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(win.index, mlat, label="CDMAG_MLAT", color="tab:orange", linewidth=1.0, linestyle="--")
        ax1_twin.set_ylabel("CDMAG_MLAT (deg)", fontsize=11, color="tab:orange")
        ax1_twin.tick_params(axis='y', labelcolor="tab:orange")
    
    ax1.grid(True, alpha=0.2)
    ax1.set_title(f"{spacecraft} MagEphem Context - {date_str}", fontsize=12, fontweight="bold")
    
    # Panel 2: CDMAG_MLT
    if "CDMAG_MLT" in win.columns:
        mlt = pd.to_numeric(win["CDMAG_MLT"], errors="coerce")
        ax2.plot(win.index, mlt, label="CDMAG_MLT", color="tab:green", linewidth=1.2)
        ax2.set_ylabel("MLT (hours)", fontsize=11)
        ax2.set_ylim(0, 24)
    
    ax2.grid(True, alpha=0.2)
    
    # Panel 3: |B| + BoverBeq
    bmag = None
    if all(c in win.columns for c in ("Bsc_gsm_0", "Bsc_gsm_1", "Bsc_gsm_2")):
        bx = pd.to_numeric(win["Bsc_gsm_0"], errors="coerce")
        by = pd.to_numeric(win["Bsc_gsm_1"], errors="coerce")
        bz = pd.to_numeric(win["Bsc_gsm_2"], errors="coerce")
        bmag = np.sqrt(bx**2 + by**2 + bz**2)
        ax3.plot(win.index, bmag, label="|B|", color="tab:red", linewidth=1.2)
        ax3.set_ylabel("|B| (nT)", fontsize=11, color="tab:red")
        ax3.tick_params(axis='y', labelcolor="tab:red")
    
    if "BoverBeq" in win.columns:
        boveq = pd.to_numeric(win["BoverBeq"], errors="coerce")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(win.index, boveq, label="BoverBeq", color="tab:purple", linewidth=1.0, linestyle="--")
        ax3_twin.set_ylabel("B/Beq", fontsize=11, color="tab:purple")
        ax3_twin.tick_params(axis='y', labelcolor="tab:purple")
    
    ax3.grid(True, alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax3.set_xlabel("UTC", fontsize=11)
    
    # Add burst time markers (thin vertical lines) to all panels
    if burst_times is not None and len(burst_times) > 0:
        burst_times_dt = pd.to_datetime(burst_times, utc=True, errors='coerce')
        burst_times_dt = burst_times_dt[burst_times_dt.notna()]
        
        for bt in burst_times_dt:
            ax1.axvline(bt, color='red', linewidth=0.8, alpha=0.5, linestyle='-', zorder=5)
            ax2.axvline(bt, color='red', linewidth=0.8, alpha=0.5, linestyle='-', zorder=5)
            ax3.axvline(bt, color='red', linewidth=0.8, alpha=0.5, linestyle='-', zorder=5)
    
    for ax in fig.get_axes():
        ax.tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    
    plt.show()

def plot_spacecraft_locations_meridional_fig(
    magephem_df: pd.DataFrame,
    start_utc: str,
    end_utc: str,
    spacecraft="RBSP-A",
    event_utc: str | None = None,
    plot_Lshells=(2.0, 3.0, 4.0),
    xlim=(0.5, 6.5),
    ylim=(-2.5, 2.5),
):
    """
    Standalone figure version: plot spacecraft locations from MagEphem in a
    meridional-plane proxy (creates its own figure).
      x = sqrt(Rsm_0^2 + Rsm_1^2)  (Re)
      z = Rsm_2                    (Re)
    """
    df = _magephem_index(magephem_df)

    t0 = pd.to_datetime(start_utc, utc=True)
    t1 = pd.to_datetime(end_utc, utc=True)
    win = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
    if win.empty:
        raise ValueError("No MagEphem samples in requested time window.")

    x = np.sqrt(win["Rsm_0"].to_numpy()**2 + win["Rsm_1"].to_numpy()**2)
    z = win["Rsm_2"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.add_patch(Circle((0,0), 1.0, fill=False, lw=2))
    r_iono = 1.0 + 90.0/RE_KM
    ax.add_patch(Circle((0,0), r_iono, fill=False, lw=1))

    L_handles = []
    for L in plot_Lshells:
        xl, zl = _dipole_fieldline(float(L))
        h, = ax.plot(xl, zl, "--", lw=1)
        L_handles.append((h, f"L = {L:g}"))
    ax.plot(x, z, ".", markersize=2, label=f"{spacecraft} track")

    if event_utc is not None:
        te = pd.to_datetime(event_utc, utc=True)
        r0_e = _interp_magephem_col(win, "Rsm_0", te)
        r1_e = _interp_magephem_col(win, "Rsm_1", te)
        r2_e = _interp_magephem_col(win, "Rsm_2", te)
        if r0_e is not None and r1_e is not None and r2_e is not None:
            x_e   = float(np.sqrt(r0_e**2 + r1_e**2))
            z_e   = float(r2_e)
            alt_km = (float(np.sqrt(r0_e**2 + r1_e**2 + r2_e**2)) - 1.0) * RE_KM
        else:
            i = win.index.get_indexer([te], method="nearest")[0]
            x_e, z_e = x[i], z[i]
            row = win.iloc[i]
            alt_km = (np.sqrt(row["Rsm_0"]**2 + row["Rsm_1"]**2 + row["Rsm_2"]**2) - 1.0) * RE_KM
        ax.plot([x_e], [z_e], marker="*", markersize=14, label="event")

        meta_ev = sample_magephem_at_time(win, te)
        Ls   = meta_ev.get("Lsimple")
        mlat = meta_ev.get("CDMAG_MLAT")
        mlt  = meta_ev.get("CDMAG_MLT")

        title = f"{spacecraft}  {te.strftime('%d %b %Y %H:%M:%S UT')}  Alt={alt_km:,.0f} km"
        if Ls   is not None and np.isfinite(Ls):   title += f"  L={Ls:.2f}"
        if mlat is not None and np.isfinite(mlat): title += f"  λm={mlat:.1f}°"
        if mlt  is not None and np.isfinite(mlt):  title += f"  MLT={mlt:.1f}"
        ax.set_title(title)
    else:
        ax.set_title(f"{spacecraft}  {t0.strftime('%d %b %Y')} {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')} UT")

    ax.set_xlabel(r"$R_E$")
    ax.set_ylabel(r"$R_E$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    handles, labels = zip(*L_handles)
    ax.legend(handles, labels, loc="upper left", title="L-shells")
    plt.tight_layout()
    plt.show()

def plot_emfsis_waveform_broken_x(
    times_records,
    waveform_2d,
    fs: float,
    spacecraft: str,
    component: str = "Bw",
    fmax_hz: float = 6000.0,
    vmin: float = -13,
    vmax: float = -7,
    cmap: str = "turbo",
    gap_threshold_s: float = 20.0,
    max_panels: int = 10,
    figsize_per_panel=(3.8, 4.2),
    single_record_width_ratio: float = 0.6,
    outpath: str | None = None,
):

    t_rec = pd.to_datetime(times_records, utc=True)
    w = np.asarray(waveform_2d)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    Nrec, Nsamp = w.shape
    rec_dur = Nsamp / fs
    order = np.argsort(t_rec.values)
    t_rec = t_rec[order]
    w = w[order, :]

    dt = np.diff(t_rec.values).astype("timedelta64[ns]").astype(np.int64) / 1e9
    breaks = np.where(dt > (rec_dur + gap_threshold_s))[0] + 1
    starts = np.r_[0, breaks]
    ends   = np.r_[breaks, Nrec]
    clusters = list(zip(starts, ends))
    if len(clusters) > max_panels:
        clusters = clusters[:max_panels-1] + [(clusters[max_panels-1][0], clusters[-1][1])]
    n_panels = len(clusters)

    all_psd_values = []
    for i in range(Nrec):
        w_sel = np.asarray(w[i], dtype=float).flatten()
        f_hz, t_sec, psd = signal.spectrogram(
            w_sel,
            fs=fs,
            nperseg=1024,
            noverlap=1000,
            nfft=1024,
            scaling="density",
            mode="psd",
        )
        
        mask = (f_hz >= 0) & (f_hz <= fmax_hz)
        if np.any(mask):
            Z = np.maximum(psd[mask, :], 1e-30)
            Z_log = np.log10(Z)
            all_psd_values.extend(Z_log.flatten())
    
    if all_psd_values:
        global_vmin = np.nanpercentile(all_psd_values, 5)
        global_vmax = np.nanpercentile(all_psd_values, 99.5)
    else:
        global_vmin, global_vmax = vmin, vmax

    width_ratios = []
    for (i0, i1) in clusters:
        n = i1 - i0
        width_ratios.append(single_record_width_ratio if n == 1 else 1.0)

    fig_w = figsize_per_panel[0] * sum(width_ratios)
    fig_h = figsize_per_panel[1]

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=False
    )
    if n_panels == 1:
        axes = [axes]

    im = None
    for pi, (i0, i1) in enumerate(clusters):
        ax = axes[pi]
        n_in_panel = i1 - i0

        t0 = t_rec[i0]
        t1 = t_rec[i1-1] + pd.to_timedelta(rec_dur, unit="s")
        span = (t1 - t0).total_seconds()

        for i in range(i0, i1):
            w_sel = np.asarray(w[i], dtype=float).flatten()
            f_hz, t_sec, psd = signal.spectrogram(
                w_sel,
                fs=fs,
                nperseg=1024,
                noverlap=1000,
                nfft=1024,
                scaling="density",
                mode="psd",
            )

            mask = (f_hz >= 0) & (f_hz <= fmax_hz)
            if not np.any(mask):
                continue

            Z = np.maximum(psd[mask, :], 1e-30)
            Zlog = np.log10(Z)

            x0 = (t_rec[i] - t0).total_seconds()
            x = x0 + t_sec
            y = f_hz[mask] / 1000.0

            im = ax.pcolormesh(x, y, Zlog, shading="auto", cmap=cmap, vmin=global_vmin, vmax=global_vmax)

        ax.set_ylim(0.01, fmax_hz / 1000.0)
        ax.set_xlim(0, max(span, rec_dur))
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{t0.strftime('%H:%M:%S')} UT (N={n_in_panel})", fontsize=10)

        d = 0.015
        if pi > 0:
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.3)
            ax.plot((-d, +d), (1-d, 1+d), **kwargs)
            ax.plot((-d, +d), (-d, +d), **kwargs)
        if pi < n_panels - 1:
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.3)
            ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)

    axes[0].set_ylabel("Frequency (kHz)")

    fig.suptitle(f"EMFISIS Waveform Spectrogram ({spacecraft} {component})",fontsize=18, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.12, top=0.88, wspace=0.05)
    if im is not None:
        cax = fig.add_axes((0.90, 0.12, 0.025, 0.76))
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")

        ticks = [-13, -11, -9, -7]
        ticks = [tv for tv in ticks if global_vmin <= tv <= global_vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([rf"$10^{{{int(tv)}}}$" for tv in ticks])
        cbar.set_label(f"{component} PSD (nT²/Hz)", rotation=270, labelpad=15)

    plt.show()

def plot_emfsis_mlat_combined(
    spectral_times,
    spectral_freqs_hz,
    spectral_log10,
    magephem_df: pd.DataFrame,
    t_start_utc,
    t_end_utc,
    spacecraft: str = "RBSP-A",
    burst_times=None,
    burst_intervals=None,   # list of (t_start, t_end) pairs for axvspan shading
    hfr_df: pd.DataFrame | None = None,   # optional HFR density DataFrame for f_pe overlay
    root_dir: str | None = None,           # if set and hfr_df is None, auto-load HFR density
    outpath: str | None = None,
):
    if hfr_df is None and root_dir is not None:
        try:
            _t0_str = pd.to_datetime(t_start_utc, utc=True).strftime("%Y-%m-%d %H:%M:%S")
            _t1_str = pd.to_datetime(t_end_utc,   utc=True).strftime("%Y-%m-%d %H:%M:%S")
            hfr_df = get_hfr_density_df(spacecraft, _t0_str, _t1_str, root_dir)
            if hfr_df.empty:
                hfr_df = None
        except Exception as _hfr_exc:
            print(f"[WARN] HFR auto-load failed: {_hfr_exc}")
            hfr_df = None

    t_spec = pd.to_datetime(spectral_times, utc=True, errors="coerce")
    ok_t = t_spec.notna()
    t_spec = t_spec[ok_t]

    freqs_hz = np.asarray(spectral_freqs_hz, dtype=float)
    Z = np.asarray(spectral_log10, dtype=float)

    if Z.shape[0] != len(pd.to_datetime(spectral_times, utc=True, errors="coerce")):
        if Z.shape[0] != len(t_spec) and Z.shape[1] == len(t_spec):
            Z = Z.T

    if Z.shape[0] != len(t_spec):
        if Z.shape[0] == ok_t.shape[0]:
            Z = Z[ok_t, :]
        elif Z.shape[1] == ok_t.shape[0]:
            Z = Z[:, ok_t].T
        else:
            raise ValueError(f"Spectral shape mismatch: Z={Z.shape}, t_spec={len(t_spec)}")

    t0 = pd.to_datetime(t_start_utc, utc=True, errors="coerce")
    t1 = pd.to_datetime(t_end_utc, utc=True, errors="coerce")
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("Invalid t_start_utc / t_end_utc")

    m_win = (t_spec >= t0) & (t_spec <= t1)
    if not np.any(m_win):
        raise ValueError("No spectral samples in the requested time window")

    t_spec = t_spec[m_win]
    Z = Z[m_win, :]

    f_khz = freqs_hz / 1e3


    mdf = _magephem_index(magephem_df)
    if mdf is None or mdf.empty:
        raise ValueError("magephem_df is empty or has no valid timestamps")

    mdf_win = mdf.loc[(mdf.index >= t0) & (mdf.index <= t1)].copy()
    if mdf_win.empty:
        mdf_win = mdf.copy()

    burst_dt = None
    if burst_times is not None:
        burst_dt = pd.to_datetime(burst_times, utc=True, errors="coerce")
        burst_dt = burst_dt[burst_dt.notna()]
        if len(burst_dt) == 0:
            burst_dt = None

    fig = plt.figure(figsize=(14.5, 8.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[2.25, 1.0],
        height_ratios=[3.0, 1.05, 1.0],
        top=0.93, bottom=0.07,
        wspace=0.30, hspace=0.35
    )

    ax_spec  = fig.add_subplot(gs[0, 0])
    ax_storm = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_fce   = fig.add_subplot(gs[2, 0], sharex=ax_spec)

    gs_right = gs[0:3, 1].subgridspec(2, 1, hspace=0.0)
    ax_orbit = fig.add_subplot(gs_right[0], projection="polar")
    ax_meridional = fig.add_subplot(gs_right[1])

    ax_meridional.set_box_aspect(1)


    fig.canvas.draw()
    p_storm = ax_storm.get_position()
    p_fce   = ax_fce.get_position()
    ax_fce.set_position((p_fce.x0, p_storm.y0 - p_fce.height, p_fce.width, p_fce.height))

    im = ax_spec.pcolormesh(t_spec, f_khz, Z.T, shading="auto")
    ax_spec.set_ylabel("Frequency (kHz)")
    ax_spec.set_ylim(0.01, np.nanmax(f_khz))

    header = f"{spacecraft}   {t_spec[0].strftime('%d %b %Y  %H:%M:%S UT')}"
    try:
        t0_hdr = t_spec[0]
        meta = sample_magephem_at_time(mdf, t0_hdr)
        rsm0 = _interp_magephem_col(mdf, "Rsm_0", t0_hdr)
        rsm1 = _interp_magephem_col(mdf, "Rsm_1", t0_hdr)
        rsm2 = _interp_magephem_col(mdf, "Rsm_2", t0_hdr)
        alt = None
        if rsm0 is not None and rsm1 is not None and rsm2 is not None:
            alt = (float(np.sqrt(rsm0**2 + rsm1**2 + rsm2**2)) - 1.0) * RE_KM

        parts = [spacecraft, t0_hdr.strftime("%d %b %Y %H:%M:%S UT")]
        if alt is not None:
            parts.append(f"Altitude = {alt:,.0f} km")
        if meta.get("Lsimple") is not None:
            parts.append(f"L = {meta['Lsimple']:.2f}")
        if meta.get("CDMAG_MLAT") is not None:
            parts.append(f"λm = {meta['CDMAG_MLAT']:.1f}°")
        for cand in ("CDMAG_MLON", "CDMAG_MLON360", "CDMAG_GLON", "CDMAG_LON"):
            v = _interp_magephem_col(mdf, cand, t0_hdr)
            if v is not None:
                parts.append(f"ϕm = {v:.1f}°")
                break
        if meta.get("CDMAG_MLT") is not None:
            parts.append(f"MLT = {meta['CDMAG_MLT']:.1f}")
        header = "   ".join(parts)
    except Exception:
        pass

    fig.suptitle("EMFISIS Spectral Data", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.955, header, fontsize=10, ha="center", va="top", color="0.25")

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.046)
    cbar.set_label("log₁₀(B² nT²/Hz)")

    plot_storm_context_from_magephem(
        ax_storm,
        magephem_df,
        tmin=t0.strftime("%Y-%m-%d %H:%M:%S"),
        tmax=t1.strftime("%Y-%m-%d %H:%M:%S"),
    )
    ax_storm.set_xlabel("")

    plot_fce_context_from_magephem(
        ax_fce,
        magephem_df,
        tmin=t0.strftime("%Y-%m-%d %H:%M:%S"),
        tmax=t1.strftime("%Y-%m-%d %H:%M:%S"),
        b_prefix="Bsc_gsm",
        hfr_df=hfr_df,
        title="",
        legend_fontsize=8,
        legend_framealpha=0.85,
    )
    ax_fce.set_title("")


    _intervals = None
    if burst_intervals is not None:
        _intervals = []
        for pair in burst_intervals:
            ts = pd.to_datetime(pair[0], utc=True, errors="coerce")
            te = pd.to_datetime(pair[1], utc=True, errors="coerce")
            if not pd.isna(ts) and not pd.isna(te):
                _intervals.append((ts, te))
        if len(_intervals) == 0:
            _intervals = None

    if _intervals is not None:
        for ts, te in _intervals:
            ax_spec.axvline(ts, color="red", linewidth=0.9, alpha=0.7, linestyle="--", zorder=6)
            ax_spec.axvline(te, color="red", linewidth=0.9, alpha=0.7, linestyle="--", zorder=6)

            for ax in (ax_storm, ax_fce):
                ax.axvspan(ts, te, color="red", alpha=0.10, linewidth=0, zorder=2)
                ax.axvline(ts, color="red", linewidth=0.8, alpha=0.6, linestyle="--", zorder=3)
                ax.axvline(te, color="red", linewidth=0.8, alpha=0.6, linestyle="--", zorder=3)
    elif burst_dt is not None:
        for bt in burst_dt:
            for ax in (ax_spec, ax_storm, ax_fce):
                ax.axvline(bt, color="red", linewidth=0.8, alpha=0.5, linestyle="-", zorder=5)

    if ("CDMAG_MLT" in mdf_win.columns) and ("Lsimple" in mdf_win.columns):
        mlt  = pd.to_numeric(mdf_win["CDMAG_MLT"], errors="coerce")
        L    = pd.to_numeric(mdf_win["Lsimple"], errors="coerce")
        mlat = pd.to_numeric(mdf_win["CDMAG_MLAT"], errors="coerce") if "CDMAG_MLAT" in mdf_win.columns else None

        valid = mlt.notna() & L.notna()
        if mlat is not None:
            valid = valid & mlat.notna()

        if valid.any():
            mlt_v  = mlt[valid].to_numpy()
            L_v    = L[valid].to_numpy()
            mlat_v = mlat[valid].to_numpy() if mlat is not None else None

            theta = 2.0 * np.pi * (mlt_v / 24.0)
            ax_orbit.set_theta_zero_location("N")  # type: ignore[attr-defined]
            ax_orbit.set_theta_direction(-1)  # type: ignore[attr-defined]

            earth_r = 1.0
            
            earth_disk = Circle((0, 0), earth_r, transform=ax_orbit.transData._b,  # type: ignore[attr-defined]
                              fill=True, facecolor="0.85", edgecolor="0.4", linewidth=1.5, zorder=0)
            ax_orbit.add_patch(earth_disk)
            
            night_wedge = Wedge(
                (0, 0), earth_r,
                theta1=0.0,
                theta2=180.0,
                transform=ax_orbit.transData._b,  # type: ignore[attr-defined]
                facecolor="0.35",
                alpha=0.6,
                edgecolor="none",
                zorder=1
            )
            ax_orbit.add_patch(night_wedge)
            
            day_wedge = Wedge(
                (0, 0), earth_r,
                theta1=180.0,
                theta2=360.0,
                transform=ax_orbit.transData._b,  # type: ignore[attr-defined]
                facecolor="1.0",
                alpha=0.15,
                edgecolor="none",
                zorder=1
            )
            ax_orbit.add_patch(day_wedge)

            if mlat_v is not None:
                sc = ax_orbit.scatter(theta, L_v, c=mlat_v, cmap="coolwarm", s=8, alpha=0.8, edgecolor="none")
                cb = fig.colorbar(sc, ax=ax_orbit, pad=0.12, fraction=0.12, orientation='horizontal')
                cb.set_label("CDMAG_MLAT (deg)")
            else:
                ax_orbit.scatter(theta, L_v, s=8, alpha=0.8, edgecolor="none")

            for Lref in [2, 3, 4, 5, 6]:
                circ = Circle((0, 0), Lref, transform=ax_orbit.transData._b,  # type: ignore[attr-defined]
                              fill=False, edgecolor="0.7", linestyle="--", linewidth=0.8, alpha=0.7)
                ax_orbit.add_artist(circ)


            ax_orbit.set_rmin(0)  # type: ignore[attr-defined]
            ax_orbit.set_rmax(max(6, np.nanmax(L_v) * 1.05))  # type: ignore[attr-defined]

            ax_orbit.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax_orbit.set_xticklabels(["00\n(Midnight)", "03", "06\n(Dawn)", "09",
                                      "12\n(Noon)", "15", "18\n(Dusk)", "21"])


            if burst_dt is not None:
                for bt in burst_dt:
                    if bt < mdf_win.index.min() or bt > mdf_win.index.max():
                        continue
                    try:
                        b_mlt = _interp_magephem_col(mdf_win, "CDMAG_MLT", bt)
                        b_L   = _interp_magephem_col(mdf_win, "Lsimple", bt)
                        if b_mlt is None or b_L is None:
                            continue
                        th = 2.0 * np.pi * (b_mlt / 24.0)
                        ax_orbit.plot([th], [b_L], marker="o", markersize=6,
                                      color="red", alpha=0.9, zorder=10,
                                      markeredgecolor="white", markeredgewidth=0.6)
                    except Exception:
                        pass
        else:
            ax_orbit.text(0.5, 0.5, "No valid MLT/L in window",
                          transform=ax_orbit.transAxes, ha="center", va="center")
    else:
        ax_orbit.text(0.5, 0.5, "Missing CDMAG_MLT or Lsimple",
                      transform=ax_orbit.transAxes, ha="center", va="center")

    try:
        plot_spacecraft_locations_meridional(
            ax_meridional,
            magephem_df,
            start_utc=t0.strftime("%Y-%m-%d %H:%M:%S"),
            end_utc=t1.strftime("%Y-%m-%d %H:%M:%S"),
            spacecraft=spacecraft,
            burst_times=burst_dt,
            plot_Lshells=(2.0, 3.0, 4.0),
        )
    except Exception as e:
        ax_meridional.text(0.5, 0.5, f"Meridional plot error:\n{str(e)[:50]}",
                           transform=ax_meridional.transAxes, ha="center", va="center", fontsize=9)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.DateFormatter("%H:%M")

    for ax in (ax_spec, ax_storm, ax_fce):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", rotation=0)
        ax.margins(x=0)
        ax.grid(False)

    ax_spec.tick_params(axis="x", which="both", labelbottom=True)
    ax_storm.tick_params(axis="x", which="both", labelbottom=False)
    ax_fce.tick_params(axis="x", which="both", labelbottom=True)

    ax_spec.set_xlabel("UTC")
    ax_storm.set_xlabel("")
    ax_fce.set_xlabel("UTC")

    plt.setp(ax_spec.get_xticklabels(), visible=True)
    plt.setp(ax_fce.get_xticklabels(), visible=True)

    plt.show()

# ====================================================================================================
# Support Functions
# ====================================================================================================

def _dipole_fieldline(L: float, n=400):
    lam = np.linspace(-np.deg2rad(75), np.deg2rad(75), n)
    r = L * (np.cos(lam) ** 2)
    x = r * np.cos(lam)
    z = r * np.sin(lam)
    return x, z

def mlt_to_wedge_deg(mlt):
    """Convert MLT (Magnetic Local Time) to Matplotlib Wedge degrees."""
    return (90.0 - 15.0 * mlt) % 360.0

def _fmt_ts(dt) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def _magephem_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        return out.sort_index()

    if "IsoTime" in df.columns:
        t = pd.to_datetime(df["IsoTime"], utc=True, errors="coerce")
        out = df.copy()
        out["_t"] = t
        out = out.dropna(subset=["_t"]).set_index("_t").sort_index()
        return out

    # fallback: try parse current index
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna().sort_index()
    return out

def _interp_magephem_col(df: pd.DataFrame, col: str, t: pd.Timestamp) -> float | None:
    """Linearly interpolate a single MagEphem column to timestamp t.
    df must already have a UTC DatetimeIndex (call _magephem_index first).
    """
    if col not in df.columns:
        return None
    try:
        t_unix = df.index.astype("int64").to_numpy() / 1e9
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        v = float(np.interp(t.timestamp(), t_unix, vals))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _nearest_row_by_utc(mage_df: pd.DataFrame, utc_dt) -> tuple[pd.Series | None, int | None]:
    if mage_df is None or len(mage_df) == 0:
        return None, None

    dt = pd.to_datetime(utc_dt, utc=True, errors="coerce")
    if pd.isna(dt):
        return None, None

    if "UTC" in mage_df.columns:
        utc_num = pd.to_numeric(mage_df["UTC"], errors="coerce")
        if utc_num.notna().any():
            med = float(np.nanmedian(np.abs(utc_num.to_numpy(dtype=float))))
            if med > 1e15:
                target = dt.value  # ns
            elif med > 1e12:
                target = dt.value / 1e6  # ms
            elif med > 1e9:
                target = dt.value / 1e9  # s
            else:
                target = dt.value / 1e9  # default to s

            diffs = np.abs(utc_num.to_numpy(dtype=float) - target)
            if np.isfinite(diffs).any():
                idx = int(np.nanargmin(diffs))
                return mage_df.iloc[idx], idx

    if "IsoTime" in mage_df.columns:
        t = pd.to_datetime(mage_df["IsoTime"], utc=True, errors="coerce")
        if t.notna().any():
            diffs = np.abs((t - dt).to_numpy(dtype="timedelta64[ns]").astype(np.int64))
            idx = int(np.nanargmin(diffs))
            return mage_df.iloc[idx], idx

    return None, None

def sample_magephem_at_time(mage_df: pd.DataFrame, t_iso_or_datetime):
    if mage_df is None or len(mage_df) == 0:
        return {}

    dt = pd.to_datetime(t_iso_or_datetime, utc=True, errors="coerce")
    if pd.isna(dt):
        return {}

    # Normalise to a UTC DatetimeIndex
    df = _magephem_index(mage_df)
    if df is None or df.empty:
        return {}

    t_unix = df.index.astype("int64").to_numpy() / 1e9
    t_sec  = dt.timestamp()

    nearest_idx = int(np.argmin(np.abs(df.index - dt)))
    row = df.iloc[nearest_idx]

    def _interp(col: str):
        """Linearly interpolate a single numeric column; return None on failure."""
        if col not in df.columns:
            return None
        try:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            v = float(np.interp(t_sec, t_unix, vals))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    result = {}

    if "IsoTime" in row.index:
        result["IsoTime"] = row["IsoTime"]

    result["Lsimple"] = _interp("Lsimple")

    result["CDMAG_MLAT"] = _interp("CDMAG_MLAT")
    result["CDMAG_MLT"]  = _interp("CDMAG_MLT")

    v_inout = _interp("InOut")
    result["InOut"] = int(round(v_inout)) if v_inout is not None else None

    bx = _interp("Bsc_gsm_0")
    by = _interp("Bsc_gsm_1")
    bz = _interp("Bsc_gsm_2")
    if bx is not None and by is not None and bz is not None:
        bmag = float(np.sqrt(bx**2 + by**2 + bz**2))
        result["Bmag"] = bmag if np.isfinite(bmag) else None
    else:
        # fallback: try pre-computed magnitude column
        result["Bmag"] = _interp("Bsc_gsm_3")

    result["BoverBeq"] = _interp("BoverBeq")

    result["Loss_Cone_Alpha_n"] = _interp("Loss_Cone_Alpha_n")
    result["Loss_Cone_Alpha_s"] = _interp("Loss_Cone_Alpha_s")

    result["RadiusOfCurv"] = _interp("RadiusOfCurv")
    result["d2B_ds2"]       = _interp("d2B_ds2")

    result["Kp"]              = _interp("Kp")
    result["Dst"]             = _interp("Dst")
    result["DipoleTiltAngle"] = _interp("DipoleTiltAngle")

    return {k: v for k, v in result.items() if v is not None}

def plot_storm_context_from_magephem(
    ax,
    magephem_df: pd.DataFrame,
    tmin: str | None = None,
    tmax: str | None = None
):
    if magephem_df is None or len(magephem_df) == 0:
        return ax, None

    df = magephem_df.copy()

    if "IsoTime" in df.columns:
        t = pd.to_datetime(df["IsoTime"], utc=True, errors="coerce")
        df = df.loc[t.notna()].copy()
        df.index = t[t.notna()]
    elif not isinstance(df.index, pd.DatetimeIndex):
        t = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[t.notna()].copy()
        df.index = t[t.notna()]
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    # Optional crop
    if tmin is not None:
        t0 = pd.to_datetime(tmin, utc=True, errors="coerce")
        if t0 is not None:
            df = df.loc[df.index >= t0]
    if tmax is not None:
        t1 = pd.to_datetime(tmax, utc=True, errors="coerce")
        if t1 is not None:
            df = df.loc[df.index <= t1]

    if df.empty:
        return ax, None

    dst = None
    kp = None
    if "Dst" in df.columns:
        dst = pd.to_numeric(df["Dst"], errors="coerce").dropna()
    if "Kp" in df.columns:
        kp = pd.to_numeric(df["Kp"], errors="coerce").dropna()

    bar_width = None
    if kp is not None and len(kp) > 1:
        dt_median = np.median(np.diff(kp.index.view('int64'))) / 1e9 / 3600.0  # hours
        bar_width = dt_median / 24.0  # Convert to matplotlib date units (days)

    # Plot Kp bars (left axis)
    if kp is not None and len(kp) > 0:
        ax.bar(kp.index, kp, width=bar_width, color="0.85", edgecolor="none", align="center")
        ax.set_ylabel("Kp", fontsize=11)
        kp_max = max(9, int(np.ceil(kp.max())))
        ax.set_ylim(0, kp_max)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)
        ax.text(0.02, 0.92, "Kp", transform=ax.transAxes, fontsize=12, color="0.3", 
                verticalalignment="top", fontweight="bold")

    # Plot Dst line (right axis)
    ax2 = ax.twinx()
    if dst is not None and len(dst) > 0:
        ax2.plot(dst.index, dst, color="k", linewidth=1.3)
        ax2.set_ylabel("Dst (nT)", fontsize=11)
        dst_min = dst.min()
        dst_max = dst.max()
        ylim_bottom = max(-150, int(np.floor(dst_min / 50)) * 50 - 10)
        ylim_top = min(50, int(np.ceil(dst_max / 50)) * 50 + 10)
        ax2.set_ylim(ylim_bottom, ylim_top)
        ax2.set_yticks([50, 0, -50, -100])
        ax2.spines["top"].set_visible(True)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)
        ax2.text(0.98, 0.92, "Dst", transform=ax2.transAxes, fontsize=12, color="k", 
                 verticalalignment="top", horizontalalignment="right", fontweight="bold")

    # X-axis formatting (concise UTC)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    
    ax.grid(False)
    ax2.grid(False)

    return ax, ax2

def _get_bmag_from_magephem(df: pd.DataFrame, b_prefix: str = "Bsc_gsm") -> pd.Series:
    
    mag_col = f"{b_prefix}_3"
    if mag_col in df.columns:
        bmag = pd.to_numeric(df[mag_col], errors="coerce")
        bmag = bmag.replace([np.inf, -np.inf], np.nan)
        if bmag.notna().any():
            return bmag

    c0, c1, c2 = f"{b_prefix}_0", f"{b_prefix}_1", f"{b_prefix}_2"
    if all(c in df.columns for c in (c0, c1, c2)):
        bx = pd.to_numeric(df[c0], errors="coerce")
        by = pd.to_numeric(df[c1], errors="coerce")
        bz = pd.to_numeric(df[c2], errors="coerce")
        bmag = pd.Series(np.where(np.isinf(np.sqrt(bx**2 + by**2 + bz**2)),
                                  np.nan,
                                  np.sqrt(bx**2 + by**2 + bz**2)),
                         index=df.index)
        return bmag

    raise KeyError(f"Could not find {mag_col} or {b_prefix}_0..2 in MagEphem columns.")

def plot_fce_context_from_magephem(
    ax,
    magephem_df: pd.DataFrame,
    tmin: str | None = None,
    tmax: str | None = None,
    b_prefix: str = "Bsc_gsm",
    hfr_df: pd.DataFrame | None = None,
    title: str = "",
    legend_fontsize: float = 9,
    legend_framealpha: float = 1.0,
):
    has_hfr = hfr_df is not None and not hfr_df.empty
    if not title:
        title = (r"Local cyclotron & plasma frequencies $f_{ce}$, $f_{pe}$"
                 if has_hfr else r"Local cyclotron frequency $f_{ce}$")
    ylabel = r"$f_{ce}$, $f_{pe}$ (kHz)" if has_hfr else r"$f_{ce}$ (kHz)"

    if magephem_df is None or len(magephem_df) == 0:
        ax.text(
            0.01, 0.5,
            "No MagEphem data available",
            transform=ax.transAxes, ha="left", va="center"
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    df = magephem_df.copy()

    # Ensure UTC DatetimeIndex
    if "IsoTime" in df.columns:
        t = pd.to_datetime(df["IsoTime"], utc=True, errors="coerce")
        df = df.loc[t.notna()].copy()
        df.index = t[t.notna()]
    elif not isinstance(df.index, pd.DatetimeIndex):
        t = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[t.notna()].copy()
        df.index = t[t.notna()]
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    if tmin is not None:
        df = df[df.index >= pd.to_datetime(tmin, utc=True)]
    if tmax is not None:
        df = df[df.index <= pd.to_datetime(tmax, utc=True)]

    if df.empty:
        ax.text(
            0.01, 0.5,
            "No data in time window",
            transform=ax.transAxes, ha="left", va="center"
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    try:
        bmag_nT = _get_bmag_from_magephem(df, b_prefix=b_prefix)
    except Exception as e:
        ax.text(
            0.01, 0.5,
            f"Cannot compute fce from MagEphem.\n{e}",
            transform=ax.transAxes, ha="left", va="center", fontsize=9
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    bmag_nT = bmag_nT.dropna()
    if bmag_nT.empty:
        ax.text(
            0.01, 0.5,
            f"{b_prefix}: |B| has no valid samples in this interval.",
            transform=ax.transAxes, ha="left", va="center", fontsize=9
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    fce_khz = _FCE_KHZ_PER_NT * bmag_nT

    ax.plot(fce_khz.index, fce_khz.values, linewidth=1.5, label=r"$f_{ce}$", color="blue")
    
    if hfr_df is not None and not hfr_df.empty:
        plot_fpe_context_from_hfr(
            ax, hfr_df, 
            tmin=str(tmin) if tmin else "", 
            tmax=str(tmax) if tmax else "",
            label=r"$f_{pe}$", 
            color="orange", 
            linewidth=1.2, 
            alpha=0.85
        )
    
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=legend_fontsize, 
                  framealpha=legend_framealpha)

    return ax

def compute_fpe_khz_from_ne_cm3(ne_cm3: np.ndarray) -> np.ndarray:
    ne = np.asarray(ne_cm3, dtype=float)
    return _FPE_KHZ_CONVERSION * np.sqrt(np.maximum(ne, 0.0))

def plot_fpe_context_from_hfr(
    ax,
    hfr_df: pd.DataFrame,
    tmin: str,
    tmax: str,
    fpe_col: str | None = None,
    ne_col: str = "Ne_cm3",
    label: str = r"$f_{pe}$",
    color: str = "orange",
    linewidth: float = 1.2,
    alpha: float = 0.9,
):
    if hfr_df is None or hfr_df.empty:
        return ax

    df = hfr_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    t0 = pd.to_datetime(tmin, utc=True, errors="coerce")
    t1 = pd.to_datetime(tmax, utc=True, errors="coerce")
    if not pd.isna(t0):
        df = df[df.index >= t0]
    if not pd.isna(t1):
        df = df[df.index <= t1]

    if df.empty:
        return ax

    fpe_khz = None
    
    if fpe_col is not None and fpe_col in df.columns:
        fpe_hz = df[fpe_col].to_numpy()
        fpe_khz = fpe_hz / 1e3
    elif "f_pe_hz" in df.columns:
        fpe_hz = df["f_pe_hz"].to_numpy()
        fpe_khz = fpe_hz / 1e3
    elif ne_col in df.columns:
        fpe_khz = compute_fpe_khz_from_ne_cm3(df[ne_col].to_numpy())
    else:
        return ax

    if fpe_khz is None:
        return ax
    
    valid = np.isfinite(fpe_khz)
    if not valid.any():
        return ax

    ax.plot(
        df.index[valid], fpe_khz[valid],
        color=color, linewidth=linewidth, alpha=alpha,
        label=label, linestyle="-",
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.85)

    return ax

def plot_spacecraft_locations_meridional(
    ax,
    magephem_df: pd.DataFrame,
    start_utc: str,
    end_utc: str,
    spacecraft="RBSP-A",
    event_utc: str | None = None,
    burst_times=None,
    plot_Lshells=(2.0, 3.0, 4.0),
    xlim=(0.5, 6.5),
    ylim=(-2.5, 2.5),
):
    df = _magephem_index(magephem_df)

    t0 = pd.to_datetime(start_utc, utc=True)
    t1 = pd.to_datetime(end_utc, utc=True)
    win = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
    if win.empty:
        raise ValueError("No MagEphem samples in requested time window.")

    x = np.sqrt(win["Rsm_0"].to_numpy()**2 + win["Rsm_1"].to_numpy()**2)
    z = win["Rsm_2"].to_numpy()

    ax.add_patch(Circle((0,0), 1.0, fill=False, lw=2))
    r_iono = 1.0 + IONO_ALT_KM/RE_KM
    ax.add_patch(Circle((0,0), r_iono, fill=False, lw=1))


    L_handles = []
    for L in plot_Lshells:
        xl, zl = _dipole_fieldline(float(L))
        h, = ax.plot(xl, zl, "--", lw=1, alpha=0.6, color="0.5")
        L_handles.append((h, f"L = {L:g}"))
    
    mlat = None
    if "CDMAG_MLAT" in win.columns:
        mlat = pd.to_numeric(win["CDMAG_MLAT"], errors="coerce").to_numpy()
    
    if mlat is not None and np.isfinite(mlat).any():
        sc = ax.scatter(x, z, c=mlat, cmap="coolwarm", s=8, alpha=0.8, edgecolor="none", label=f"{spacecraft} track")
    else:
        ax.plot(x, z, ".", markersize=2, label=f"{spacecraft} track", color="C0", alpha=0.7)


    if event_utc is not None:
        te = pd.to_datetime(event_utc, utc=True)
        r0_e = _interp_magephem_col(win, "Rsm_0", te)
        r1_e = _interp_magephem_col(win, "Rsm_1", te)
        r2_e = _interp_magephem_col(win, "Rsm_2", te)
        if r0_e is not None and r1_e is not None and r2_e is not None:
            x_e = float(np.sqrt(r0_e**2 + r1_e**2))
            z_e = float(r2_e)
        else:
            i = win.index.get_indexer([te], method="nearest")[0]
            x_e, z_e = x[i], z[i]
        ax.plot([x_e], [z_e], marker="*", markersize=14, color="red", label="event", zorder=10)


    if burst_times is not None:
        burst_dt = pd.to_datetime(burst_times, utc=True, errors="coerce")
        burst_dt = burst_dt[burst_dt.notna()]
        if len(burst_dt) > 0:
            for bt in burst_dt:
                if bt < win.index.min() or bt > win.index.max():
                    continue
                try:
                    r0 = _interp_magephem_col(win, "Rsm_0", bt)
                    r1 = _interp_magephem_col(win, "Rsm_1", bt)
                    r2 = _interp_magephem_col(win, "Rsm_2", bt)
                    if r0 is None or r1 is None or r2 is None:
                        continue
                    b_x = float(np.sqrt(r0**2 + r1**2))
                    b_z = float(r2)
                    ax.plot([b_x], [b_z], marker="o", markersize=6,
                            color="red", alpha=0.9, zorder=10,
                            markeredgecolor="white", markeredgewidth=0.6)
                except Exception:
                    pass

    ax.set_xlabel(r"$R_E$")
    ax.set_ylabel(r"$R_E$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_title("Meridional Plane", fontsize=10, fontweight="bold")
    
    if L_handles:
        handles, labels = zip(*L_handles)
        ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)
