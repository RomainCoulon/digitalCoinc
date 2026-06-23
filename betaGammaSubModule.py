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
    data_bg  : dict with beta singles rate (B), gamma rate (G), beta-gamma
               coincidence count rate (BG), their uncertainties, and live/real time
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
    enlarge_window  = config["settings"].getfloat("enlarge_window")
    n_run           = config["settings"].getint("n_run_tdcr")
    delay = (lowerDelayBound+upperDelayBound)/2
    lowerDelayBound -= delay
    upperDelayBound -= delay
    lowerDelayBound *= enlarge_window
    upperDelayBound *= enlarge_window
    
    # ── State variables ────────────────────────────────────────────────────────
    beta_trigger  = 0.0    # time of last beta dead-time trigger
    beta_dtW      = 0.0    # beta dead-time window duration
    gamma_trigger = 0.0    # time of last gamma dead-time trigger
    gamma_dtW     = 0.0    # gamma dead-time window duration
 
    # Pending accepted events — reset after a coincidence or when window expires
    pending_beta  = None   # (timetag, energy) of last accepted beta,  or None
    pending_gamma = None   # (timetag, energy) of last accepted gamma, or None
 
    BGcount       = 0
    Bcount        = 0   # accepted beta (D/T above threshold) events in current run
    Gcount        = 0   # accepted gamma (in-ROI) events in current run
    liveTime      = 0.0   # beta live time (driven by beta triggers)
    liveTime_gamma = 0.0  # gamma live time (driven by gamma triggers)
    realTime      = 0.0
    w             = 1
 
    BGrate       = []
    Brate        = []
    Grate        = []
    lt           = []   # beta live times per run
    lt_gamma     = []   # gamma live times per run
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
                        Brate.append(Bcount / liveTime)
                        Grate.append(Gcount / liveTime)
                        lt.append(liveTime)
                        lt_gamma.append(liveTime_gamma)
                    BGcount       = 0
                    Bcount        = 0
                    Gcount        = 0
                    liveTime      = 0.0
                    liveTime_gamma = 0.0
                    w += 1
 
                # (3) Accept beta if above threshold
                if (channel in ("D", "T")) and (charge > thres_i):
                    Bcount += 1
                    pending_beta = (t_evt, charge)
 
                    # Check for a pending gamma within the delay window
                    # dt = t_beta - t_gamma: positive if beta after gamma, negative if before
                    if pending_gamma is not None:
                        dt = t_evt - pending_gamma[0]
                        # print(dt)
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
                # (G.1) Gamma live-time update
                liveTime_gamma += t_evt - (gamma_trigger + gamma_dtW)

                # Accept gamma if inside energy window
                if lowerBoundGamma <= charge <= upperBoundGamma:
                    Gcount += 1
                    pending_gamma = (t_evt, charge)

                    # Check for a pending beta within the delay window
                    # dt = t_beta - t_gamma: positive if beta after gamma, negative if before
                    if pending_beta is not None:
                        dt = pending_beta[0] - t_evt
                        # print(dt)
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
        Brate.append(Bcount / liveTime)
        Grate.append(Gcount / liveTime)
        lt.append(liveTime)
        lt_gamma.append(liveTime_gamma)

    # ── Accidental coincidence correction (per run, then average) ──────────────
    # Live-time fractions per run (paper notation):
    #   T_β = t_l^(β) / t_r
    #   T_γ = t_l^(γ) / t_r
    #   T_C = T_β × T_γ   (coincidence channel live when both are simultaneously live)
    # Coincidence window half-width used as r_γ = r_β = T_C (digital framework).
    # τ_β = beta_dead_time, τ_γ = gamma_dead_time.
    #
    # Accidental rates (eq. 9):
    #
    #   R_f^(γ→β) = R_β · T_C · (R_γ - R_βγ) / (R_βγ · T_β²)
    #               · [1 - (T_C / (T_β · T_γ))^(-r_γ / τ_β)]
    #
    #   R_f^(β→γ) = R_γ · (R_β - R_βγ) / (T_γ · R_βγ) · T_C / T_γ
    #               · [1 + R_βγ·r_β - R_βγ·max(0, r_β - τ_γ + τ_β)
    #                  - (T_C / (T_β · T_γ))^(-max(0, r_β - τ_γ + τ_β) / τ_β)]
    #
    # Iterative solution: R_βγ = R_C - R_f^(γ→β) - R_f^(β→γ)

    def _accidental_corr_run(R_C, R_b, R_g, T_b, T_g, tau_b, tau_g,
                             n_iter=30, tol=1e-9):
        """Return (R_bg_corr, R_acc_gb, R_acc_bg, n_iter_done) for one run."""
        T_c = T_b * T_g   # coincidence live-time fraction
        r   = T_c         # r_γ = r_β = T_C (digital framework)

        def _rates(R_bg):
            if R_bg <= 0 or T_b <= 0 or T_g <= 0 or T_c <= 0:
                return 0.0, 0.0

            base = T_c / (T_b * T_g)   # = T_g (always, but keep symbolic)

            # ── γ→β ──────────────────────────────────────────────────────────
            exp_gb = -r / tau_b
            try:
                bracket_gb = 1.0 - base ** exp_gb
            except (OverflowError, ValueError, ZeroDivisionError):
                bracket_gb = 1.0

            R_gb = (R_b * T_c * (R_g - R_bg)) / (R_bg * T_b**2) * bracket_gb

            # ── β→γ ──────────────────────────────────────────────────────────
            m    = max(0.0, r - tau_g + tau_b)   # max(0, r_β - τ_γ + τ_β)
            exp_bg = -m / tau_b if tau_b > 0 else 0.0
            try:
                power_bg = base ** exp_bg
            except (OverflowError, ValueError, ZeroDivisionError):
                power_bg = 1.0

            bracket_bg = 1.0 + R_bg * r - R_bg * m - power_bg

            R_bg_term = (R_g * (R_b - R_bg)) / (T_g * R_bg) * (T_c / T_g) * bracket_bg

            return R_gb, R_bg_term

        R_bg = R_C   # initial guess
        R_acc_gb = R_acc_bg_val = 0.0

        for k in range(n_iter):
            R_acc_gb, R_acc_bg_val = _rates(R_bg)
            R_bg_new = R_C - R_acc_gb - R_acc_bg_val
            if abs(R_bg_new - R_bg) < tol:
                R_bg = R_bg_new
                return R_bg, R_acc_gb, R_acc_bg_val, k + 1
            R_bg = R_bg_new

        print(f"  [accidental correction] Warning: did not converge in {n_iter} iterations "
              f"(residual = {abs(R_bg_new - R_bg):.2e})")
        return R_bg, R_acc_gb, R_acc_bg_val, n_iter

    BGrate_corr   = []
    R_acc_gb_list = []
    R_acc_bg_list = []

    for R_C_i, R_b_i, R_g_i, lt_b_i, lt_g_i in zip(BGrate, Brate, Grate, lt, lt_gamma):
        T_b_i = lt_b_i / realTime
        T_g_i = lt_g_i / realTime if realTime > 0 else 0.0

        R_bg_corr_i, r_gb_i, r_bg_i, nit = _accidental_corr_run(
            R_C=R_C_i, R_b=R_b_i, R_g=R_g_i,
            T_b=T_b_i, T_g=T_g_i,
            tau_b=beta_dead_time, tau_g=gamma_dead_time,
        )
        BGrate_corr.append(R_bg_corr_i)
        R_acc_gb_list.append(r_gb_i)
        R_acc_bg_list.append(r_bg_i)

    print(f"  Accidental correction (γ→β): {np.mean(R_acc_gb_list):.6f} s⁻¹  "
          f"(β→γ): {np.mean(R_acc_bg_list):.6f} s⁻¹  "
          f"total: {np.mean(R_acc_gb_list)+np.mean(R_acc_bg_list):.6f} s⁻¹")
    print(f"  BG raw:       {np.mean(BGrate):.6f} ± {np.std(BGrate):.6f} s⁻¹")
    print(f"  BG corrected: {np.mean(BGrate_corr):.6f} ± {np.std(BGrate_corr):.6f} s⁻¹")
 
    # ── Results ────────────────────────────────────────────────────────────────
    data_bg = {
        "thres_i":              thres_i,
        "lowerBoundGamma":      lowerBoundGamma,
        "upperBoundGamma":      upperBoundGamma,
        "lowerDelayBound_ns":   lowerDelayBound * 1e9,
        "upperDelayBound_ns":   upperDelayBound * 1e9,
        "B":                    np.mean(Brate),
        "u_B":                  np.std(Brate),
        "G":                    np.mean(Grate),
        "u_G":                  np.std(Grate),
        # Raw coincidence rate
        "BG":                   np.mean(BGrate),
        "u_BG":                 np.std(BGrate),
        # Accidental-corrected coincidence rate
        "BG_corr":              np.mean(BGrate_corr),
        "u_BG_corr":            np.std(BGrate_corr),
        # Individual accidental contributions
        "R_acc_gamma_to_beta":  np.mean(R_acc_gb_list),
        "R_acc_beta_to_gamma":  np.mean(R_acc_bg_list),
        "R_acc_total":          np.mean(R_acc_gb_list) + np.mean(R_acc_bg_list),
        # Per-run arrays (raw)
        "B_runs":               np.array(Brate),
        "G_runs":               np.array(Grate),
        "BG_runs":              np.array(BGrate),
        # Per-run arrays (corrected)
        "BG_corr_runs":         np.array(BGrate_corr),
        "Dead_time_percent":    round(100 - 100 * sum(lt) / realTime, 1),
        "Live_time_s":          lt,
        "Live_time_gamma_s":    lt_gamma,
        "Real_time_s":          realTime,
    }
 
    df_coinc = pd.DataFrame(
        coinc_events,
        columns=["TIMETAG_beta", "TIMETAG_gamma", "ENERGY_beta", "ENERGY_gamma"]
    )
 
    print(f"beta-gamma coincidence events: {len(df_coinc)}")
    
    # Add to bg_process flush block:
    print(f"  lt_beta / lt_gamma ratio: {sum(lt)/sum(lt_gamma):.4f}")
 
    return data_bg, df_coinc