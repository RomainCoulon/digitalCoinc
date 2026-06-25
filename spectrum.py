# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:37:36 2026

@author: romain.coulon
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser

def energySpectrum(df, bins=1024, label="Channel", Bounds=None):
    """
    Plot the energy spectrum of a channel.
 
    Parameters
    ----------
    df     : DataFrame with an ENERGY column
    bins   : number of ADC bins (default 1024)
    label  : plot title suffix
    Bounds : None, or a (lower, upper) tuple of ADC channel values.
             When provided, draws vertical lines and shades the selected region.
    """
    counts, edges = np.histogram(df["ENERGY"], bins=bins, range=(0, bins))
    centers = (edges[:-1] + edges[1:]) / 2
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(centers, counts, where="mid", linewidth=0.8)
 
    if Bounds is not None:
        lower, upper = Bounds
        ax.axvline(lower, color="red",   linestyle="--", linewidth=1.0, label=f"Lower = {lower}")
        ax.axvline(upper, color="green", linestyle="--", linewidth=1.0, label=f"Upper = {upper}")
        ax.axvspan(lower, upper, alpha=0.1, color="yellow", label="ROI")
        ax.legend()
 
    ax.set_xlabel("Energy (ADC channels)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Energy Spectrum — {label}")
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
 
    return counts, edges

def timeDiffSpectrum(dfB, dfG, Bounds=None, window=1e-6, bin_width=1e-9, label="β–γ", energyBoundsG=None):
    """
    Compute and plot the time difference distribution between beta and gamma events.
 
    For each gamma event, find all beta events within ±window (seconds) and
    record dt = t_beta - t_gamma.
 
    Parameters
    ----------
    dfB           : DataFrame with columns TIMETAG (s, float64) for beta
    dfG           : DataFrame with columns TIMETAG (s, float64) and ENERGY for gamma
    window        : half-width of the coincidence search window in seconds (default 1 µs)
    bin_width     : histogram bin width in seconds (default 1 ns)
    label         : plot title suffix
    energyBoundsG : None, or a (lower, upper) tuple of ADC channel values. When
                    provided, only gamma events with ENERGY within this range
                    (the gamma ROI) are used to build the time-difference
                    spectrum; gamma events outside the ROI are discarded.
    """
    if energyBoundsG is not None:
        eLowG, eUpG = energyBoundsG
        n_before = len(dfG)
        dfG = dfG[(dfG["ENERGY"] >= eLowG) & (dfG["ENERGY"] <= eUpG)]
        print(f"Gamma events in ROI for time-diff spectrum: {len(dfG)} / {n_before}")

    t_beta  = dfB["TIMETAG"].to_numpy(dtype="float64")
    t_gamma = dfG["TIMETAG"].to_numpy(dtype="float64")
 
    # Sort both arrays (required for searchsorted)
    t_beta  = np.sort(t_beta)
    t_gamma = np.sort(t_gamma)
 
    # Collect all dt = t_beta - t_gamma within ±window
    dt_list = []
    for tg in t_gamma:
        lo = np.searchsorted(t_beta, tg - window, side="left")
        hi = np.searchsorted(t_beta, tg + window, side="right")
        if hi > lo:
            dt_list.append(t_beta[lo:hi] - tg)
 
    if not dt_list:
        print("No coincidences found in the given window.")
        return None, None
 
    dt = np.concatenate(dt_list)
 
    # Histogram
    bins  = np.arange(-window, window + bin_width, bin_width)
    counts, edges = np.histogram(dt, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(centers * 1e9, counts, where="mid", linewidth=0.8)   # display in ns
    
    if Bounds is not None:
        lower, upper = Bounds
        ax.axvline(lower, color="red",   linestyle="--", linewidth=1.0, label=f"Lower = {lower}")
        ax.axvline(upper, color="green", linestyle="--", linewidth=1.0, label=f"Upper = {upper}")
        ax.axvspan(lower, upper, alpha=0.1, color="yellow", label="ROI")
        ax.legend()
    
    ax.set_xlabel("$t_{\\beta} - t_{\\gamma}$ (ns)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Time Difference Spectrum — {label}")
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$\\Delta t = 0$")
    ax.grid(True, which="both", alpha=0.3)
    # ax.legend()
    plt.tight_layout()
    plt.show()

    return counts, edges


def sesam_spectrum(t_beta_accepted, t_gamma_roi, T_dead, bin_width=500e-9,
                   coinc_exclusion=0.0, label=""):
    """
    Plot the SESAM time-difference spectrum (Müller 1981 Fig. 2) with symmetric guards.

    Zone layout (relative to each accepted beta t₀, displayed in µs):

      [−2T+exc, −T−exc]  → Plateau G  (blue)   — all gammas, same width as g
      [−T−exc,  −T+exc]  → Guard      (red)     — dead-time boundary exclusion
      [−T+exc,  −exc]    → Gap g      (orange)  — only unpaired gammas
      [−exc,    0]       → Guard      (red)     — coincidence peak exclusion

    Parameters
    ----------
    t_beta_accepted : sorted array of accepted beta timestamps (s)
    t_gamma_roi     : sorted array of gamma timestamps in energy ROI (s)
    T_dead          : extended dead time (s)
    bin_width       : histogram bin width in seconds (default 500 ns)
    coinc_exclusion : guard width on each edge of g and G (s)
    label           : plot title suffix
    """
    exc       = coinc_exclusion
    search_lo = -(2.0 * T_dead + exc)   # show full G zone plus a margin
    search_hi =  0.3 * T_dead

    dt_list = []
    tg = np.asarray(t_gamma_roi)
    for t0 in t_beta_accepted:
        lo = np.searchsorted(tg, t0 + search_lo, side='left')
        hi = np.searchsorted(tg, t0 + search_hi, side='right')
        if hi > lo:
            dt_list.append(tg[lo:hi] - t0)

    if not dt_list:
        print("sesam_spectrum: no gamma events found in the SESAM window.")
        return None, None

    dt_us = np.concatenate(dt_list) * 1e6

    bw_us = bin_width * 1e6
    bins  = np.arange(search_lo * 1e6, search_hi * 1e6 + bw_us, bw_us)
    counts, edges = np.histogram(dt_us, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    T_us   = T_dead * 1e6
    exc_us = exc * 1e6
    W_us   = T_us - 2.0 * exc_us     # effective zone width in µs

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(centers, counts, where='mid', linewidth=0.8, color='steelblue')

    # Plateau G
    ax.axvspan(-2*T_us + exc_us, -T_us - exc_us, alpha=0.15, color='blue',
               label=f'Plateau G  (W = {W_us*1e3:.0f} ns)')
    # Dead-time boundary guard
    ax.axvspan(-T_us - exc_us, -T_us + exc_us, alpha=0.30, color='red',
               label=f'Guard ±{exc_us*1e3:.0f} ns (DT boundary)')
    # Gap g
    ax.axvspan(-T_us + exc_us, -exc_us, alpha=0.15, color='orange',
               label=f'Gap g  (W = {W_us*1e3:.0f} ns)')
    # Coincidence peak guard
    ax.axvspan(-exc_us, 0, alpha=0.30, color='red',
               label=f'Guard ±{exc_us*1e3:.0f} ns (coinc. peak)')

    for x, c, ls in [(-2*T_us+exc_us, 'blue',   ':'),
                     (-T_us-exc_us,   'red',    '--'),
                     (-T_us+exc_us,   'red',    '--'),
                     (-exc_us,        'orange', '--'),
                     (0,              'grey',   ':')]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=1.0)

    ax.set_xlabel(r'$t_{\gamma} - t_{\beta_0}$  (µs)')
    ax.set_ylabel('Counts per bin')
    ax.set_title(f'SESAM time spectrum{" — " + label if label else ""}  '
                 f'[T = {T_us:.0f} µs, guard = {exc_us*1e3:.0f} ns]')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return counts, edges


def autoDetectDelayBounds(dfB, dfG, window=3e-6, bin_width=2e-9,
                           energyBoundsG=None, n_sigma=2.0, plot=True, label="β–γ"):
    """
    Automatically determine the beta-gamma delay bounds from the time-difference spectrum.

    Algorithm:
      1. Build the time-difference histogram (dt = t_beta - t_gamma) over ±window.
      2. Estimate the flat accidental background from the outer 10 % of bins on each side.
      3. Smooth the histogram with a ~10 ns running average to reduce noise.
      4. Locate the peak (maximum of the smoothed histogram).
      5. Walk left and right from the peak until the smoothed counts drop below
         background + n_sigma * std(background).

    Parameters
    ----------
    dfB           : DataFrame with TIMETAG column (beta channel)
    dfG           : DataFrame with TIMETAG and ENERGY columns (gamma channel)
    window        : half-width of the search range in seconds (default 3 µs)
    bin_width     : histogram bin width in seconds (default 2 ns)
    energyBoundsG : (lower, upper) ADC channel range to pre-filter gamma events
    n_sigma       : threshold above background for bound detection (default 2)
    plot          : whether to plot the result (default True)
    label         : plot title suffix

    Returns
    -------
    lower_ns, upper_ns : detected bounds in nanoseconds
    peak_ns            : peak position in nanoseconds
    """
    if energyBoundsG is not None:
        eLowG, eUpG = energyBoundsG
        dfG = dfG[(dfG["ENERGY"] >= eLowG) & (dfG["ENERGY"] <= eUpG)]

    t_beta  = np.sort(dfB["TIMETAG"].to_numpy(dtype="float64"))
    t_gamma = np.sort(dfG["TIMETAG"].to_numpy(dtype="float64"))

    # Build time-difference histogram
    dt_list = []
    for tg in t_gamma:
        lo = np.searchsorted(t_beta, tg - window, side="left")
        hi = np.searchsorted(t_beta, tg + window, side="right")
        if hi > lo:
            dt_list.append(t_beta[lo:hi] - tg)

    if not dt_list:
        raise RuntimeError("autoDetectDelayBounds: no coincidences found within the search window.")

    dt = np.concatenate(dt_list)           # raw dt values in seconds — kept for median
    bins   = np.arange(-window, window + bin_width, bin_width)
    counts, edges = np.histogram(dt, bins=bins)
    centers_ns = (edges[:-1] + edges[1:]) / 2 * 1e9  # convert to ns

    # Estimate background from outer 10 % of bins on each side
    n_outer = max(1, int(0.10 * len(counts)))
    bg_samples = np.concatenate([counts[:n_outer], counts[-n_outer:]])
    background     = np.mean(bg_samples)
    background_std = np.std(bg_samples)
    threshold = background + n_sigma * background_std

    # Smooth with a running average (~10 ns window)
    smooth_bins = max(1, int(10e-9 / bin_width))
    kernel = np.ones(smooth_bins) / smooth_bins
    counts_smooth = np.convolve(counts, kernel, mode="same")

    # Rough peak location via argmax (biased by tails — used only for walking)
    rough_idx = int(np.argmax(counts_smooth))

    if counts_smooth[rough_idx] <= threshold:
        raise RuntimeError(
            f"autoDetectDelayBounds: peak ({counts_smooth[rough_idx]:.0f}) is not above "
            f"the detection threshold ({threshold:.0f}). Check the search window or energy ROI."
        )

    # Walk left from rough peak to find lower bound
    left_idx = rough_idx
    while left_idx > 0 and counts_smooth[left_idx] > threshold:
        left_idx -= 1

    # Walk right from rough peak to find upper bound
    right_idx = rough_idx
    while right_idx < len(counts_smooth) - 1 and counts_smooth[right_idx] > threshold:
        right_idx += 1

    lower_ns = centers_ns[left_idx]
    upper_ns = centers_ns[right_idx]

    # Robust delay estimator: median of raw dt values inside the detected bounds.
    # The median is insensitive to the right tail that biases the argmax and the
    # midpoint-of-bounds estimators.
    dt_in_peak = dt[(dt * 1e9 >= lower_ns) & (dt * 1e9 <= upper_ns)]
    peak_ns = float(np.median(dt_in_peak) * 1e9) if len(dt_in_peak) > 0 \
              else centers_ns[rough_idx]

    print(f"\nauto-detected delay bounds: [{lower_ns:.1f}, {upper_ns:.1f}] ns")
    print(f"  delay estimator (median) : {peak_ns:.1f} ns")
    print(f"  rough peak (argmax)      : {centers_ns[rough_idx]:.1f} ns")
    print(f"  window width             : {upper_ns - lower_ns:.1f} ns")
    print(f"  background               : {background:.1f} ± {background_std:.1f} counts/bin")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(centers_ns, counts, where="mid", linewidth=0.8, label="raw histogram")
        ax.plot(centers_ns, counts_smooth, linewidth=1.2, color="orange", label="smoothed")
        ax.axhline(background, color="gray",   linestyle=":",  linewidth=1.0, label=f"background = {background:.0f}")
        ax.axhline(threshold,  color="purple", linestyle="--", linewidth=1.0, label=f"threshold ({n_sigma}σ) = {threshold:.0f}")
        ax.axvline(lower_ns, color="red",   linestyle="--", linewidth=1.2, label=f"lower = {lower_ns:.1f} ns")
        ax.axvline(upper_ns, color="green", linestyle="--", linewidth=1.2, label=f"upper = {upper_ns:.1f} ns")
        ax.axvline(peak_ns,  color="blue",  linestyle="-",  linewidth=1.5, label=f"median delay = {peak_ns:.1f} ns")
        ax.axvspan(lower_ns, upper_ns, alpha=0.12, color="yellow", label="auto ROI")
        ax.set_xlabel(r"$t_{\beta} - t_{\gamma}$ (ns)")
        ax.set_ylabel("Counts")
        ax.set_title(f"Time Difference Spectrum — auto-detected bounds [{label}]")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return lower_ns, upper_ns, peak_ns