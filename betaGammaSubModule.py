# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:34:26 2026

@author: romain.coulon
"""

import configparser
import numpy as np
import pandas as pd

def mergeBG(df_out, df_beta, df_G):
    """
    Merge out-of-interst events (S or S and D), beta events D or T, and gamma event into a single time-sorted DataFrame.
 
    
    Returns
    -------
    df_bg : merged and time-sorted DataFrame with an added 'CHANNEL' column
    """
    
    df_out["CHANNEL"]="O"
        
    df_bg = pd.concat([df_out, df_beta, df_G], ignore_index=True)
    df_bg.sort_values("TIMETAG", inplace=True)
    df_bg.reset_index(drop=True, inplace=True)
 
    print(f"Merged beta shape: {df_bg.shape}")
    return df_bg
    
    
    
    

def gammaExtDTprocess(df_G, confileFileName="config.ini"):
    """
    Live-time processing for the gamma channel (single channel, NaI detector).
 
    For each event above threshold, an extended dead time (gamma_dead_time) is
    imposed. Only events whose ENERGY falls within [lowerBoundGamma, upperBoundGamma]
    are counted in the gamma rate.
 
    Parameters
    ----------
    df_G            : time-sorted DataFrame with columns TIMETAG (s, float64) and ENERGY
    confileFileName : path to config.ini
 
    Returns
    -------
    data_gamma : dict of count rate, uncertainty, live/real time and dead-time fraction
    df_gamma   : DataFrame of accepted gamma events (within energy bounds), with
                 columns TIMETAG, ENERGY, CHANNEL='G'
    """
    config = configparser.ConfigParser()
    config.read(confileFileName)
    ExtDT          = config["settings"].getfloat("gamma_dead_time")   * 1e-9   # ns → s
    lowerBoundGamma = config["settings"].getfloat("lowerBoundGamma")
    upperBoundGamma = config["settings"].getfloat("upperBoundGamma")
    n_run          = config["settings"].getint("n_run_tdcr")
 
    # ── Counter and time initialisation ───────────────────────────────────────
    Gcount       = 0
    trigger_time = 0.0   # time of the last accepted trigger
    comdtW       = 0.0   # current dead-time window end (relative to trigger_time)
    realTime     = 0.0
    liveTime     = 0.0
    w            = 1
 
    Grate = []
    lt    = []
    gamma_events = []   # (timetag, energy) of accepted in-window gamma events
 
    t_mes = df_G["TIMETAG"].iloc[-1] / n_run
 
    # ── Main loop ──────────────────────────────────────────────────────────────
    for row in df_G.itertuples(index=False):
        t_evt  = row.TIMETAG
        charge = row.ENERGY
 
        realTime = t_evt
 
        # ── (1) EVENT ARRIVES WHEN SYSTEM IS IDLE ─────────────────────────────
        if t_evt > trigger_time + comdtW:
 
            # (1.1) Intermediate result at end of each run
            if realTime > t_mes * w:
                print(f"gamma run #{w}  {int(realTime / 60)} min processed.")
                if liveTime > 0:
                    Grate.append(Gcount / liveTime)
                    lt.append(liveTime)
                Gcount   = 0
                liveTime = 0.0
                w += 1
 
            # (1.2) Update live time
            liveTime += t_evt - (trigger_time + comdtW)
 
            # (1.3) Count event if inside energy window
            if lowerBoundGamma <= charge <= upperBoundGamma:
                Gcount += 1
                gamma_events.append((t_evt, charge))
 
            # (1.4) Set new trigger and impose dead time
            trigger_time = t_evt
            comdtW       = ExtDT
 
        # ── (2) EVENT ARRIVES WHEN SYSTEM IS BUSY → ignored (pure extended DT) ─
 
    # ── Flush last run ─────────────────────────────────────────────────────────
    if liveTime > 0:
        print(f"gamma run #{w}  {int(realTime / 60)} min processed. [last]")
        Grate.append(Gcount / liveTime)
        lt.append(liveTime)
 
    n_actual = len(Grate)
 
    # ── Results ────────────────────────────────────────────────────────────────
    data_gamma = {
        "lowerBoundGamma":    lowerBoundGamma,
        "upperBoundGamma":    upperBoundGamma,
        "G":                  np.mean(Grate),
        "u_G":                np.std(Grate),
        "Dead_time_percent":  round(100 - 100 * sum(lt) / realTime, 1),
        "Live_time_s":        lt,
        "Real_time_s":        realTime,
    }
 
    df_gamma = pd.DataFrame(gamma_events, columns=["TIMETAG", "ENERGY"])
    df_gamma["CHANNEL"] = "G"
 
    print(f"Gamma events in ROI: {len(df_gamma)}")
 
    return data_gamma, df_gamma 




def bg_process(df_bg, thres_i, confileFileName="config.ini"):
    """
    Process beta-gamma coincidence events and estimate the beta-gamma coincidence
    count rate using a live-time algorithm with independent dead times per channel.
 
    Channel conventions in df_bg
    -----------------------------
    "O" : out-of-interest beta event  → beta dead time applied, not counted
    "D" : double beta event           → beta dead time applied, counted if ENERGY > thres_i
    "T" : triple beta event           → beta dead time applied, counted if ENERGY > thres_i
    "G" : gamma event                 → gamma dead time applied,
                                        counted if lowerBoundGamma < ENERGY < upperBoundGamma
 
    Dead-time / coincidence logic
    ------------------------------
    Each channel has its own extended dead time (OR gate blocks the coincidence channel).
    When a beta is accepted, it opens a coincidence search: if a gamma was accepted
    within [lowerDelayBound, upperDelayBound] ns before OR after, a coincidence is recorded.
    Symmetrically, when a gamma is accepted it checks the pending beta.
 
    Parameters
    ----------
    df_bg          : time-sorted DataFrame with TIMETAG (s), ENERGY, CHANNEL columns
    thres_i        : beta energy threshold (ADC channels)
    confileFileName: path to config.ini
 
    Returns
    -------
    data_bg  : dict with coincidence count rate, uncertainty, live/real time
    df_coinc : DataFrame of accepted beta-gamma coincidence pairs
    """
    config = configparser.ConfigParser()
    config.read(confileFileName)
    beta_dead_time  = config["settings"].getfloat("com_ext_deadTime_tdcr") * 1e-9  # ns → s
    gamma_dead_time = config["settings"].getfloat("gamma_dead_time")        * 1e-9  # ns → s
    lowerBoundGamma = config["settings"].getfloat("lowerBoundGamma")
    upperBoundGamma = config["settings"].getfloat("upperBoundGamma")
    lowerDelayBound = config["settings"].getfloat("lowerDelayBound")        * 1e-9  # ns → s
    upperDelayBound = config["settings"].getfloat("upperDelayBound")        * 1e-9  # ns → s
    n_run           = config["settings"].getint("n_run_tdcr")
 
    # ── State variables ────────────────────────────────────────────────────────
    beta_trigger  = 0.0    # time of last beta dead-time trigger
    beta_dtW      = 0.0    # beta dead-time window duration
    gamma_trigger = 0.0    # time of last gamma dead-time trigger
    gamma_dtW     = 0.0    # gamma dead-time window duration
 
    # Pending accepted events — reset after a coincidence or when window expires
    pending_beta  = None   # (timetag, energy) of last accepted beta,  or None
    pending_gamma = None   # (timetag, energy) of last accepted gamma, or None
 
    BGcount  = 0
    liveTime = 0.0
    realTime = 0.0
    w        = 1
 
    BGrate       = []
    lt           = []
    coinc_events = []   # (timetag_beta, timetag_gamma, energy_beta, energy_gamma)
 
    t_mes = df_bg["TIMETAG"].iloc[-1] / n_run
 
    # ── Main loop ──────────────────────────────────────────────────────────────
    for row in df_bg.itertuples(index=False):
        t_evt   = row.TIMETAG
        charge  = row.ENERGY
        channel = row.CHANNEL
 
        realTime = t_evt
 
        beta_busy  = t_evt <= beta_trigger  + beta_dtW
        gamma_busy = t_evt <= gamma_trigger + gamma_dtW
        coinc_busy = beta_busy or gamma_busy   # OR gate
 
        # ══ BETA EVENT (O, D, T) ══════════════════════════════════════════════
        if channel in ("O", "D", "T"):
 
            if not beta_busy:
                # (1) Live time update
                liveTime += t_evt - (beta_trigger + beta_dtW)
 
                # (2) Intermediate result
                if realTime > t_mes * w:
                    print(f"bg run #{w}  {int(realTime / 60)} min processed.")
                    if liveTime > 0:
                        BGrate.append(BGcount / liveTime)
                        lt.append(liveTime)
                    BGcount  = 0
                    liveTime = 0.0
                    w += 1
 
                # (3) Accept beta if above threshold
                if (channel in ("D", "T")) and (charge > thres_i):
                    pending_beta = (t_evt, charge)
 
                    # Check for a pending gamma within the delay window
                    # dt = t_beta - t_gamma: positive if beta after gamma, negative if before
                    if not coinc_busy and pending_gamma is not None:
                        dt = t_evt - pending_gamma[0]
                        if lowerDelayBound <= dt <= upperDelayBound:
                            BGcount += 1
                            coinc_events.append((t_evt, pending_gamma[0], charge, pending_gamma[1]))
                            pending_beta  = None   # consume both
                            pending_gamma = None
 
                # (4) Expire pending gamma when beta arrived too late to ever match
                if pending_gamma is not None:
                    dt = t_evt - pending_gamma[0]   # t_beta - t_gamma
                    if dt > upperDelayBound:
                        pending_gamma = None
 
                # (5) Impose beta dead time
                beta_trigger = t_evt
                beta_dtW     = beta_dead_time
 
        # ══ GAMMA EVENT ═══════════════════════════════════════════════════════
        elif channel == "G":
 
            if not gamma_busy:
                # Accept gamma if inside energy window
                if lowerBoundGamma < charge < upperBoundGamma:
                    pending_gamma = (t_evt, charge)
 
                    # Check for a pending beta within the delay window
                    # dt = t_beta - t_gamma: positive if beta before gamma, negative if after
                    if not coinc_busy and pending_beta is not None:
                        dt = pending_beta[0] - t_evt   # same sign convention: t_beta - t_gamma
                        if lowerDelayBound <= dt <= upperDelayBound:
                            BGcount += 1
                            coinc_events.append((pending_beta[0], t_evt, pending_beta[1], charge))
                            pending_beta  = None   # consume both
                            pending_gamma = None
 
                # Expire pending beta when gamma arrived too late to ever match
                if pending_beta is not None:
                    dt = pending_beta[0] - t_evt   # t_beta - t_gamma
                    if dt < lowerDelayBound:
                        pending_beta = None
 
                # Impose gamma dead time
                gamma_trigger = t_evt
                gamma_dtW     = gamma_dead_time
 
    # ── Flush last run ─────────────────────────────────────────────────────────
    if liveTime > 0:
        print(f"bg run #{w}  {int(realTime / 60)} min processed. [last]")
        BGrate.append(BGcount / liveTime)
        lt.append(liveTime)
 
    # ── Results ────────────────────────────────────────────────────────────────
    data_bg = {
        "thres_i":           thres_i,
        "lowerBoundGamma":   lowerBoundGamma,
        "upperBoundGamma":   upperBoundGamma,
        "lowerDelayBound_ns": lowerDelayBound * 1e9,
        "upperDelayBound_ns": upperDelayBound * 1e9,
        "BG":                np.mean(BGrate),
        "u_BG":              np.std(BGrate),
        "Dead_time_percent": round(100 - 100 * sum(lt) / realTime, 1),
        "Live_time_s":       lt,
        "Real_time_s":       realTime,
    }
 
    df_coinc = pd.DataFrame(
        coinc_events,
        columns=["TIMETAG_beta", "TIMETAG_gamma", "ENERGY_beta", "ENERGY_gamma"]
    )
 
    print(f"beta-gamma coincidence events: {len(df_coinc)}")
 
    return data_bg, df_coinc