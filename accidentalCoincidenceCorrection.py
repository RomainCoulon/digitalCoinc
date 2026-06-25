# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:05:48 2025
Modified to include List-Mode Shift & Anticoincidence methods.

@author: romain.coulon
"""

import numpy as np
try:
    from numba import njit
except ImportError:
    print("WARNING: Numba is not installed. The shift method will be exceptionally slow without it.")
    def njit(func): return func # Fallback decorator if numba is missing

# =============================================================================
# ORIGINAL ANALYTICAL TDCR CORRECTION (Retained for backward compatibility)
# =============================================================================
def accidentalCoincCorr(rA,rB,rC,rAB,rBC,rAC,rD,rT,t_W):
    """This function corrects the counting data from accidental coincidence counting rates in TDCR measurement.
    
    Reference:
    C.Dutsov, P.Cassette, B.Sabot et al. Nuclear Inst. and Methods in Physics Research, A 977 (2020) 164292
    """
    
    # Calculation of uncorrelated count rates
    pA=rA-rAC-rAB+rT 
    pB=rB-rAB-rBC+rT 
    pC=rC-rAC-rBC+rT 
    pAB=rAB-rT 
    pBC=rBC-rT 
    pAC=rAC-rT 
    pS=pA+pB+pC 
    pD=pAB+pBC+pAC 
    pT=rT 

    # Calculation of accidental coincidence counting rates
    # aAB=(2*(pA*pB+pA*pBC+pB*pAC+pAC*pBC)+(pS+pD-pAB)*(pAB+pT))*t_W*1e-9 
    # aBC=(2*(pB*pC+pB*pAC+pC*pAB+pAB*pAC)+(pS+pD-pBC)*(pBC+pT))*t_W*1e-9 
    # aAC=(2*(pA*pC+pA*pBC+pC*pAB+pAB*pBC)+(pS+pD-pAC)*(pAC+pT))*t_W*1e-9 
    # aD=aAB+aBC+aAC
    # aT=(3*pA*pB*pC+pA*pBC+pB*pAC+pC*pAB+2*(pAB*pBC+pAB*pAC+pAC*pBC)+(pS+pD)*(pT))*t_W*1e-9 
    
    tw = t_W * 1e-9  # resolving time in seconds
    aAB = (2*(pA*pB + pA*pBC + pB*pAC + pAC*pBC) + (pS+pD-pAB)*(pAB+pT)) * tw
    aBC = (2*(pB*pC + pB*pAC + pC*pAB + pAB*pAC) + (pS+pD-pBC)*(pBC+pT)) * tw
    aAC = (2*(pA*pC + pA*pBC + pC*pAB + pAB*pBC) + (pS+pD-pAC)*(pAC+pT)) * tw
    aD  = aAB + aBC + aAC
    aT  = (  (pA*pBC + pB*pAC + pC*pAB
           + 2*(pAB*pBC + pAB*pAC + pAC*pBC)
           + (pS+pD)*pT) * tw
       + 3*pA*pB*pC * tw**2
      )
    
    # Calculation of corrected count rates
    rAB2=rAB-aAB
    rBC2=rBC-aBC
    rAC2=rAC-aAC
    rD2=rD-aD
    rT2=rT-aT
    
    return rAB2, rBC2, rAC2, rD2, rT2


# =============================================================================
# NEW LIST-MODE SHIFT METHOD & ANTICOINCIDENCE
# =============================================================================

@njit
def evaluate_listmode_coincidences(t_beta, t_gamma, window_lower, window_upper, shift_time=100e-6):
    """
    High-performance event matcher using the time-shift method for accidental coincidences.
    
    :param t_beta: 1D numpy array of beta timestamps (in seconds), sorted chronologically.
    :param t_gamma: 1D numpy array of gamma timestamps (in seconds), sorted chronologically.
    :param window_lower: Lower bound of coincidence window (e.g. -1e-6 s).
    :param window_upper: Upper bound of coincidence window (e.g. 1e-6 s).
    :param shift_time: Artificial delay to measure purely accidental coincidences (default 100 microseconds).
    
    :return: N_gamma, N_c_raw (raw coincidences), N_c_acc (accidental coincidences)
    """
    N_gamma = len(t_gamma)
    N_beta = len(t_beta)

    N_c_raw = 0
    N_c_acc = 0

    # Pointers allow us to avoid checking from the beginning of the beta array every time.
    # This reduces complexity from O(N*M) to O(N+M), executing in milliseconds.
    ptr_real = 0
    ptr_shift = 0

    for i in range(N_gamma):
        tg = t_gamma[i]

        # --- 1. Evaluate REAL Window ---
        # Advance pointer to the first beta that falls inside or after the lower bound
        while ptr_real < N_beta and t_beta[ptr_real] < tg + window_lower:
            ptr_real += 1
            
        # If the beta is also under the upper bound, it's a valid coincidence
        if ptr_real < N_beta and t_beta[ptr_real] <= tg + window_upper:
            N_c_raw += 1

        # --- 2. Evaluate SHIFTED Window (Accidentals) ---
        tg_shifted = tg + shift_time
        
        # Do the exact same logic on the artificially delayed gamma event
        while ptr_shift < N_beta and t_beta[ptr_shift] < tg_shifted + window_lower:
            ptr_shift += 1
            
        if ptr_shift < N_beta and t_beta[ptr_shift] <= tg_shifted + window_upper:
            N_c_acc += 1

    return N_gamma, N_c_raw, N_c_acc


def _scale_timestamps(t, duration_target, t_start_target):
    """
    Linearly rescale a sorted timestamp array so that it spans
    [t_start_target, t_start_target + duration_target].
    Used to map background-measurement timestamps into the source time domain
    before computing cross-mode SESAM spectra.
    """
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return t
    span = t[-1] - t[0]
    if span <= 0:
        return np.full_like(t, t_start_target)
    return (t - t[0]) / span * duration_target + t_start_target


def sesam_cross_correction(
    t_beta_src, t_gamma_src,
    t_beta_bkg, t_gamma_bkg,
    T_dead, duration_src, duration_bkg, coinc_exclusion=0.0
):
    """
    Cross-selective sampling background correction for SESAM.
    Implements the formula S0 = S1 - S2 - S3 + S4 of H. Liu (ICRM LSC WG, June 2026).

    The four modes are:
      S1 : all beta  (src + bkg)  ×  all gamma  (src + bkg)   ← standard SESAM on source run
      S2 : all beta  (src + bkg)  ×  gamma bkg only
      S3 : beta bkg only           ×  all gamma  (src + bkg)
      S4 : beta bkg only           ×  gamma bkg only

    S0 = S1 - S2 - S3 + S4 isolates the source-only beta-gamma association.

    For S2 and S3 (mixing timestamps from two separate runs), background timestamps
    are linearly rescaled to the source time domain so that the SESAM algorithm can
    search relative-time windows naturally.  S2 and S3 represent uncorrelated channels,
    so their SESAM spectra are flat (n_g ≈ n_G).  S4 is computed with actual background
    timestamps (scaled) and may have residual structure from environmental coincidences.

    Formula 1 (neglecting dead-time effects, Haoran slide 13):
        η_β = 1 − (N_g1 − N_g2 − N_g3 + N_g4) / (N_G1 − N_G2 − N_G3 + N_G4)

    Parameters
    ----------
    t_beta_src  : sorted accepted beta timestamps from source run (s)
    t_gamma_src : sorted gamma ROI timestamps from source run (s)
    t_beta_bkg  : sorted accepted beta timestamps from background run (s)
    t_gamma_bkg : sorted gamma ROI timestamps from background run (s)
    T_dead          : extended dead time (s)
    duration_src    : source measurement duration (s)
    duration_bkg    : background measurement duration (s)
    coinc_exclusion : guard width on each edge of gap and plateau (s)

    Returns
    -------
    dict with eps_beta_corr, u_eps_beta_corr, N_g0, N_G0,
         and the per-mode S1-S4 dicts for diagnostics
    """
    t_src_start = float(t_beta_src[0]) if len(t_beta_src) else 0.0

    # ── S1: source β × source γ  (standard SESAM) ────────────────────────────
    s1 = sesam_process(t_beta_src, t_gamma_src, T_dead, duration_src, coinc_exclusion)

    # ── Scale background timestamps into the source time domain ───────────────
    t_gamma_bkg_sc = _scale_timestamps(t_gamma_bkg, duration_src, t_src_start)
    t_beta_bkg_sc  = _scale_timestamps(t_beta_bkg,  duration_src, t_src_start)

    # ── S2: source β × background γ (expect flat: n_g2 ≈ n_G2) ──────────────
    s2 = sesam_process(t_beta_src,   t_gamma_bkg_sc, T_dead, duration_src, coinc_exclusion)

    # ── S3: background β × source γ (expect flat: n_g3 ≈ n_G3) ──────────────
    s3 = sesam_process(t_beta_bkg_sc, t_gamma_src,   T_dead, duration_src, coinc_exclusion)

    # ── S4: background β × background γ (residual background coincidences) ───
    s4 = sesam_process(t_beta_bkg_sc, t_gamma_bkg_sc, T_dead, duration_src, coinc_exclusion)

    # ── Cross-selective sampling formula ──────────────────────────────────────
    N_g0 = s1['N_g'] - s2['N_g'] - s3['N_g'] + s4['N_g']
    N_G0 = s1['N_G'] - s2['N_G'] - s3['N_G'] + s4['N_G']

    print(f"    SESAM cross-correction: N_g0={N_g0:.1f}  N_G0={N_G0:.1f}")
    print(f"      S1 N_g/N_G = {s1['N_g']}/{s1['N_G']},  "
          f"S2 {s2['N_g']}/{s2['N_G']},  "
          f"S3 {s3['N_g']}/{s3['N_G']},  "
          f"S4 {s4['N_g']}/{s4['N_G']}")

    if N_G0 > 0 and N_g0 >= 0:
        eps_corr   = max(0.0, 1.0 - N_g0 / N_G0)
        u_eps_corr = eps_corr * np.sqrt(
            max(N_g0, 1) / N_g0**2 + 1.0 / max(N_G0, 1)
        ) if N_g0 > 0 else 0.0
    else:
        eps_corr   = s1['eps_beta']
        u_eps_corr = s1['u_eps_beta']
        print("    WARNING: cross-correction denominator ≤ 0, falling back to S1.")

    return {
        "eps_beta":   eps_corr,
        "u_eps_beta": u_eps_corr,
        "N_g0": N_g0,
        "N_G0": N_G0,
        "S1": s1, "S2": s2, "S3": s3, "S4": s4,
    }


def sesam_process(t_beta_accepted, t_gamma_roi, T_dead, duration,
                  coinc_exclusion=0.0):
    """
    Digital Selective Sampling (SESAM) — Müller 1981 / Haoran list-mode implementation.

    A guard of width coinc_exclusion is removed symmetrically from BOTH edges of
    the gap g AND from BOTH edges of the plateau G, so both zones have identical
    effective widths.  The ratio N_g / N_G then directly equals (1 − ε_β) with no
    additional width normalisation.

    Zone layout relative to each accepted beta t₀:

      Upper guard (coinc. peak)  :  [t₀ − exc,          t₀)
      Gap g                      :  [t₀ − T + exc,       t₀ − exc)   width W = T − 2·exc
      Dead-time boundary guard   :  [t₀ − T − exc,       t₀ − T + exc)
      Plateau G                  :  [t₀ − 2T + exc,      t₀ − T − exc)   width W = T − 2·exc
      (Lower guard of G omitted — far from any artefact)

    ε_β = 1 − N_g / N_G          (valid because both zones have the same width W)
    N₀  = ρ_β / ε_β              where ρ_β = N_beta / duration

    Parameters
    ----------
    t_beta_accepted : sorted 1-D array of beta timestamps already through DT (s)
    t_gamma_roi     : sorted 1-D array of gamma timestamps in energy ROI (s)
    T_dead          : extended dead time applied to the beta channel (s)
    duration        : measurement duration (s)
    coinc_exclusion : guard width removed from each edge of g and G (s);
                      should be ≥ coincidence half-window after delay correction

    Returns
    -------
    dict : N_g, N_G, zone_W, eps_beta, u_eps_beta, Activity, u_Activity, R_beta
    """
    exc  = coinc_exclusion
    zone_W = T_dead - 2.0 * exc      # effective width of both g and G

    t0 = np.asarray(t_beta_accepted)
    tg = np.asarray(t_gamma_roi)

    # Gap g: [t0 - T + exc,  t0 - exc)
    hi_g = np.searchsorted(tg, t0 - exc,          side='left')
    lo_g = np.searchsorted(tg, t0 - T_dead + exc, side='left')
    N_g  = int((hi_g - lo_g).sum())

    # Plateau G: [t0 - 2T + exc,  t0 - T - exc)
    hi_G = np.searchsorted(tg, t0 - T_dead - exc,         side='left')
    lo_G = np.searchsorted(tg, t0 - 2.0 * T_dead + exc,   side='left')
    N_G  = int((hi_G - lo_G).sum())

    N_beta = len(t0)
    R_beta = N_beta / duration

    # Both zones have the same width → no normalisation needed
    if N_G > 0 and zone_W > 0:
        ratio   = N_g / N_G
        u_ratio = ratio * np.sqrt(1.0 / max(N_g, 1) + 1.0 / N_G)
    else:
        ratio, u_ratio = 1.0, 0.0

    eps_beta   = max(0.0, 1.0 - ratio)
    u_eps_beta = u_ratio

    if eps_beta > 0:
        Activity   = R_beta / eps_beta
        u_Activity = Activity * np.sqrt((u_eps_beta / eps_beta) ** 2 + 1.0 / max(N_beta, 1))
    else:
        Activity = u_Activity = 0.0

    return {
        "N_g":        N_g,
        "N_G":        N_G,
        "zone_W":     zone_W,
        "N_beta":     N_beta,
        "ratio_g_G":  ratio,
        "eps_beta":   eps_beta,
        "u_eps_beta": u_eps_beta,
        "Activity":   Activity,
        "u_Activity": u_Activity,
        "R_beta":     R_beta,
    }


def correlation_process(t_beta, t_gamma, T_interval, duration):
    """
    Correlation counting for absolute activity — Lewis, Smith & Williams (1973),
    Metrologia 9, 14-20, Case 2 (prompt beta-gamma, separate detectors).

    The measurement is divided into N = floor(duration / T_interval) equal intervals.
    For each interval i, beta (p_i) and gamma (d_i) events are counted.
    The cross-covariance is

        X = 1/(N-1) × [ Σ p_i d_i − (Σp_i)(Σd_i)/N ]

    With E[X] = c = ε_β × ε_γ × N₀ × T_interval and the mean counts per interval
    p̄ = ε_β × N₀ × T_interval,  d̄ = ε_γ × N₀ × T_interval, the activity is

        N₀ = p̄ × d̄ / (X × T_interval)

    identical in form to the B-G coincidence formula  N₀ = R_β × R_γ / R_BG.

    The statistical uncertainty follows Eq. (7) of Lewis 1973:
        σ²_X ≈ (p̄ × d̄ + X² + X) / N

    Parameters
    ----------
    t_beta     : sorted 1-D array of beta timestamps (s)
    t_gamma    : sorted 1-D array of gamma timestamps in energy ROI (s)
    T_interval : length of each counting interval (s)
    duration   : total measurement duration (s)

    Returns
    -------
    dict : N_intervals, p_mean, d_mean, X, u_X, eps_beta, u_eps_beta,
           Activity, u_Activity
    """
    if len(t_beta) == 0 or len(t_gamma) == 0 or T_interval <= 0:
        return {"Activity": 0.0, "u_Activity": 0.0,
                "eps_beta": 0.0, "u_eps_beta": 0.0, "X": 0.0, "u_X": 0.0,
                "N_intervals": 0, "p_mean": 0.0, "d_mean": 0.0}

    t_start = min(t_beta[0], t_gamma[0])
    N = int(duration / T_interval)
    if N < 3:
        raise RuntimeError(
            f"correlation_process: only {N} intervals — increase duration or reduce T_interval.")

    # Assign each event to its interval index via floor division
    p_idx = ((t_beta  - t_start) / T_interval).astype(int)
    d_idx = ((t_gamma - t_start) / T_interval).astype(int)

    # Keep only events that fall within [0, N)
    p = np.bincount(p_idx[(p_idx >= 0) & (p_idx < N)], minlength=N).astype(float)
    d = np.bincount(d_idx[(d_idx >= 0) & (d_idx < N)], minlength=N).astype(float)

    p_mean = p.mean()
    d_mean = d.mean()

    # Cross-covariance  (sample, ddof=1)
    X = (np.dot(p, d) - N * p_mean * d_mean) / (N - 1)

    # Uncertainty  σ²_X ≈ (p̄ d̄ + X² + X) / N  [Lewis 1973, Eq. 7]
    u_X = np.sqrt(max(p_mean * d_mean + X ** 2 + abs(X), 0.0) / N)

    # Activity
    if X > 0:
        N0    = p_mean * d_mean / (X * T_interval)
        u_N0  = N0 * np.sqrt(1.0 / (N * max(p_mean, 1e-12))
                             + 1.0 / (N * max(d_mean, 1e-12))
                             + (u_X / X) ** 2)
    else:
        N0 = u_N0 = 0.0

    # Beta efficiency  ε_β = X / d̄
    if d_mean > 0 and X > 0:
        eps_beta   = X / d_mean
        u_eps_beta = eps_beta * np.sqrt((u_X / X) ** 2 + 1.0 / (N * d_mean))
    else:
        eps_beta = u_eps_beta = 0.0

    return {
        "N_intervals": N,
        "T_interval":  T_interval,
        "p_mean":      p_mean,
        "d_mean":      d_mean,
        "X":           X,
        "u_X":         u_X,
        "eps_beta":    eps_beta,
        "u_eps_beta":  u_eps_beta,
        "Activity":    N0,
        "u_Activity":  u_N0,
    }


def process_anticoincidence_activity(N_beta, N_gamma, N_c_raw, N_c_acc, duration):
    """
    Evaluates activity using the Anticoincidence method and Shift-correction.
    
    In anticoincidence, we count gammas WITHOUT coincident betas.
    Some true anticoincidences are accidentally vetoed by background betas.
    The shifted coincidence count (N_c_acc) perfectly quantifies these accidental vetoes.
    
    :return: A dictionary containing absolute activity (N0) and intermediate metrics.
    """
    
    # 1. Standard Coincidence Metrics
    N_c_true = N_c_raw - N_c_acc
    
    # 2. Anticoincidence Metrics
    N_anti_raw = N_gamma - N_c_raw
    
    # Restore the gammas that were accidentally vetoed
    N_anti_true = N_anti_raw + N_c_acc
    
    # 3. Efficiency Evaluation
    # Beta efficiency evaluated via anticoincidence: eps_beta = 1 - (N_anti_true / N_gamma)
    efficiency_beta = 1.0 - (N_anti_true / N_gamma) if N_gamma > 0 else 0
    
    # 4. Activity Estimation
    # N0 = (N_beta * N_gamma) / (N_gamma - N_anti_true)
    if (N_gamma - N_anti_true) > 0:
        N0 = (N_beta * N_gamma) / (N_gamma - N_anti_true) / duration
    else:
        N0 = 0.0

    return {
        "N_beta": N_beta,
        "N_gamma": N_gamma,
        "N_c_raw": N_c_raw,
        "N_c_acc": N_c_acc,
        "N_c_true": N_c_true,
        "N_anti_raw": N_anti_raw,
        "N_anti_true": N_anti_true,
        "efficiency_beta": efficiency_beta,
        "Activity_N0": N0
    }