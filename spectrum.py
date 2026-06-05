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

def timeDiffSpectrum(dfB, dfG, Bounds=None, window=1e-6, bin_width=1e-9, label="β–γ"):
    """
    Compute and plot the time difference distribution between beta and gamma events.
 
    For each gamma event, find all beta events within ±window (seconds) and
    record dt = t_beta - t_gamma.
 
    Parameters
    ----------
    dfB       : DataFrame with columns TIMETAG (s, float64) for beta
    dfG       : DataFrame with columns TIMETAG (s, float64) for gamma
    window    : half-width of the coincidence search window in seconds (default 1 µs)
    bin_width : histogram bin width in seconds (default 1 ns)
    label     : plot title suffix
    """
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