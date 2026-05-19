# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:43:34 2026

@author: Remote
"""

from scipy.signal import spectrogram
import numpy as np
from scipy.signal import stft
import scipy.signal as signal
from scipy.ndimage import uniform_filter
import pandas as pd
import matplotlib.pyplot as plt

def plot_waveform_and_filtered(
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
    figsize=(10.5, 7.2),
    show_fce: bool = False,
    outpath: str | None = None,
    filtered_psd=None,          
    filtered_freqs_hz=None,    
    filtered_times_s=None,
    filtered_label="Filtered candidate LGW pixels",
):

    t_rec = pd.to_datetime(times_records, utc=True)

    if len(t_rec) == 0:
        print("[WARN] No waveform records available")
        return None, None

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
        return None, None

    t_plot = t_sec[t_mask]
    f_plot_khz = f_hz[f_mask] / 1e3

    Z = psd[np.ix_(f_mask, t_mask)]
    Z = np.maximum(Z, 1e-30)
    Z_log = np.log10(Z)

    # foir filtered data
    has_filtered = (
        filtered_psd is not None and
        filtered_freqs_hz is not None and
        filtered_times_s is not None
    )

    if has_filtered:
        filtered_psd = np.asarray(filtered_psd, dtype=float)
        filtered_freqs_hz = np.asarray(filtered_freqs_hz, dtype=float)
        filtered_times_s = np.asarray(filtered_times_s, dtype=float)

        # filtered_psd expected shape: (time, freq)
        if filtered_psd.shape != (len(filtered_times_s), len(filtered_freqs_hz)):
            raise ValueError(
                "filtered_psd shape must be (len(filtered_times_s), len(filtered_freqs_hz)). "
                f"Got filtered_psd={filtered_psd.shape}, "
                f"times={len(filtered_times_s)}, freqs={len(filtered_freqs_hz)}"
            )

        filt_t_mask = (filtered_times_s >= 0) & (filtered_times_s <= duration_s)
        filt_f_mask = (filtered_freqs_hz >= 0) & (filtered_freqs_hz <= fmax_hz)

        filt_t_plot = filtered_times_s[filt_t_mask]
        filt_f_plot_khz = filtered_freqs_hz[filt_f_mask] / 1e3

        # Convert from (time, freq) to (freq, time) for pcolormesh
        Zf = filtered_psd[np.ix_(filt_t_mask, filt_f_mask)].T

        # Preserve NaNs so non-detected pixels stay blank
        Zf_log = np.full_like(Zf, np.nan, dtype=float)
        good = np.isfinite(Zf) & (Zf > 0)
        Zf_log[good] = np.log10(Zf[good])
    else:
        filt_t_plot = None
        filt_f_plot_khz = None
        Zf_log = None

    # plot - subplot original and filtered
    header = (
        f"{spacecraft}  {event_dt.strftime('%d %b %Y %H:%M:%S')} UT  "
        f"(rec {idx + 1}/{w.shape[0]})"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    ax_raw, ax_filt = axes
    
    # for original
    im0 = ax_raw.pcolormesh(
        t_plot,
        f_plot_khz,
        Z_log,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax_raw.set_title(f"Raw {component} waveform spectrogram\n{header}", fontsize=10)
    ax_raw.set_ylabel("Frequency (kHz)")
    ax_raw.set_xlim(0, duration_s)
    ax_raw.set_ylim(0.01, fmax_hz / 1e3)

    cbar0 = plt.colorbar(im0, ax=ax_raw, pad=0.01, fraction=0.046)
    ticks = [-13, -11, -9, -7]
    ticks = [tv for tv in ticks if vmin <= tv <= vmax]
    cbar0.set_ticks(ticks)
    cbar0.set_ticklabels([rf"$10^{{{int(tv)}}}$" for tv in ticks])
    cbar0.set_label(rf"{component} PSD")
    
    # for filtered
    if has_filtered and Zf_log is not None:
        im1 = ax_filt.pcolormesh(
            filt_t_plot,
            filt_f_plot_khz,
            Zf_log,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        cbar1 = plt.colorbar(im1, ax=ax_filt, pad=0.01, fraction=0.046)
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels([rf"$10^{{{int(tv)}}}$" for tv in ticks])
        cbar1.set_label("Filtered PSD")

        n_detected = int(np.sum(np.isfinite(filtered_psd)))
        ax_filt.set_title(f"{filtered_label}  |  detected pixels = {n_detected}", fontsize=10)

    else:
        ax_filt.text(
            0.5,
            0.5,
            "No filtered data passed in",
            transform=ax_filt.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_filt.set_title(filtered_label, fontsize=10)

    ax_filt.set_xlabel("Time (s)")
    ax_filt.set_ylabel("Frequency (kHz)")
    ax_filt.set_xlim(0, duration_s)
    ax_filt.set_ylim(0.01, fmax_hz / 1e3)

    fig.suptitle("EMFISIS Waveform Echo Detection", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if outpath is not None:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {outpath}")

    plt.show()

    return fig, axes


##############################################################################

def means_wna_ellipticity_from_waveform(B_wave_3comp: np.ndarray, b0_hat, fs: float, 
            nperseg: int = 1024, noverlap: int = 768, nfft: int = 1024,):
    
    B_wave_3comp = np.asarray(B_wave_3comp, dtype=float)

    if B_wave_3comp.ndim != 2 or B_wave_3comp.shape[1] != 3:
        raise ValueError("B_wave_3comp must have shape (Nsamp, 3)")

    b0_hat = np.asarray(b0_hat, dtype=float)

    if b0_hat.shape != (3,):
        raise ValueError("b0_hat must have shape (3,)")

    b0_norm = np.linalg.norm(b0_hat)
    if b0_norm == 0 or not np.isfinite(b0_norm):
        raise ValueError("b0_hat has zero or invalid norm")

    b0_hat = b0_hat / b0_norm
    
    ##########################
    ##########################
    Z_components = []
    
    # (1) Complex STFT
    for k in range(3):
        freqs, times, Zxx = stft(
            B_wave_3comp[:, k],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend="constant",
            return_onesided=True,
            boundary=None,
            padded=False,
        )

        Z_components.append(Zxx.T)

    Z = np.stack(Z_components, axis=-1)  # (time, freq, component)
    
    # (2) Complex spectral matrix (3x3) <- complex STFT
    S = Z[..., :, None] * np.conj(Z[..., None, :])
    
    # Smooth up the complex spectral matrix
    time_width=3
    freq_width=3
    S_real = uniform_filter(S.real, size=(time_width, freq_width, 1, 1), mode="nearest")
    S_imag = uniform_filter(S.imag, size=(time_width, freq_width, 1, 1), mode="nearest")
    
    S = S_real + 1j*S_imag
    
    #extra
    ntime, nfreq = Z.shape[0], Z.shape[1]
    wna_deg = np.full((ntime, nfreq), np.nan)
    ellipticity = np.full((ntime, nfreq), np.nan)
    
    # (3) Loop through time-frequency bins
    for it in range(ntime):
        for jf in range(nfreq):
            S_tf = S[it, jf]
    
            if not np.all(np.isfinite(S_tf)):
                continue
    
            try:
                # Estimate Wave Normal Direction
                A = np.real(S_tf)
                A = 0.5 * (A + A.T)
                
                # use eigen vals and eigen vecs for variance direction
                e_vals, e_vecs = np.linalg.eigh(A)
                
                # min variance direction
                k_hat = e_vecs[:, np.argmin(e_vals)]
                k_hat = k_hat / np.linalg.norm(k_hat)
                
                # wave normal angle deg
                cosang = np.abs(np.dot(k_hat, b0_hat))
                cosang = np.clip(cosang, 0.0, 1.0)
                wna_deg[it, jf] = np.degrees(np.arccos(cosang))
                
                # ellipticity
                
                #unit vectors from k_hat
                ref = np.array([0.0, 0.0, 1.0])
                if np.abs(np.dot(ref, k_hat)) > 0.95:
                    ref = np.array([1.0, 0.0, 0.0])
            
                e1 = np.cross(ref, k_hat)
                e1 = e1 / np.linalg.norm(e1)
            
                e2 = np.cross(k_hat, e1)
                e2 = e2 / np.linalg.norm(e2)
                
                # projection matrix
                P = np.vstack([e1, e2])
                
                #2x2 transverse spectral matrix
                S2 = P @ S_tf @ P.T.conj()
                
                S11 = np.real(S2[0, 0])
                S22 = np.real(S2[1, 1])
                S12 = S2[0, 1]
                
                # Stokes Parameters
                I = S11 + S22
                Q = S11 - S22
                U = 2.0 * np.real(S12)
                V = -2.0 * np.imag(S12)
                
                if I <= 0 or not np.isfinite(I):
                    continue
                
                # More stable than V/I alone for partially polarized/noisy bins
                pol = np.sqrt(Q**2 + U**2 + V**2)

                if pol <= 0 or not np.isfinite(pol):
                    continue
                
                # Ellipticity angle chi:
                # sin(2chi) = V / I for fully polarized wave
                vfrac = np.clip(V / I, -1.0, 1.0)
                chi = 0.5 * np.arcsin(vfrac)
                # Axial ratio = |minor axis / major axis|
                ellip = np.abs(np.tan(chi))
                # Numerical safety
                ellipticity[it, jf] = float(np.clip(ellip, 0.0, 1.0))
    
            except Exception:
                continue
    
    return freqs, times, wna_deg, ellipticity

##############################################################################


def component_psd(x, fs, nperseg=1024, noverlap=768, nfft=1024):
    """
    Compute PSD spectrogram for one waveform component.
    Returns freqs, times, PSD with shape (time, freq).
    """
    freqs, times, Sxx = spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
        mode="psd",
    )

    # scipy returns Sxx as (freq, time), so transposing to (time, freq)
    return freqs, times, Sxx.T

def total_field_psd(field_3comp, fs, nperseg=1024, noverlap=768, nfft=1024):
    """
    field_3comp shape: (Nsamp, 3)
    returns freqs, times, total_PSD(time, freq)
    """
    psds = []

    for k in range(3):
        freqs, times, P = component_psd(
            field_3comp[:, k],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
        psds.append(P)

    P_total = psds[0] + psds[1] + psds[2]

    return freqs, times, P_total