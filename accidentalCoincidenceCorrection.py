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