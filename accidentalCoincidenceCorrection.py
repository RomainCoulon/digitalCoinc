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