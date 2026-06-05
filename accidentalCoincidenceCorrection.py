# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:05:48 2025

@author: romain.coulon
"""

def accidentalCoincCorr(rA,rB,rC,rAB,rBC,rAC,rD,rT,t_W):
    """This fonction correction the counting data from accidental coincidence counting rates in TDCR measurement.
    
    Reference:
    C.Dutsov,P.Cassette,B.Sabotetal. Nuclear Inst. and Methods in Physics Research,A 977 (2020) 164292
    https://doi.org/10.1016/j.nima.2020.164292    
            
    :param rA: count rate on channel A. 
    :type rA: float
    :param rB: count rate on channel B. 
    :type rB: float
    :param rC: count rate on channel C. 
    :type rC: float
    :param rAB: count rate on channel AB. 
    :type rAB: float
    :param rBC: count rate on channel BC. 
    :type rBC: float
    :param rAC: count rate on channel AC. 
    :type rAC: float
    :param rD: count rate on double coincidence D. 
    :type rD: float
    :param rT: count rate on double coincidence T. 
    :type rT: float
    :param t_W: resolving time in ns. 
    :type t_W: float

    :param rAB2: corrected count rate on channel AB. 
    :type rAB2: float
    :param rBC2: corrected count rate on channel BC. 
    :type rBC2: float
    :param rAC2: corrected count rate on channel AC. 
    :type rAC2: float
    :param rD2: corrected count rate on double coincidence D. 
    :type rD2: float
    :param rT2: corrected count rate on double coincidence T. 
    :type rT2: float


    :return: rAB2, rBC2, rAC2, rD2, rT
    :rtype: tuple of floats     
    """
    
    # Calculation of uncorrelated count rates
    pA=rA-rAC-rAB+rT # uncorrelated count rate on channel A
    pB=rB-rAB-rBC+rT # uncorrelated count rate on channel B
    pC=rC-rAC-rBC+rT # uncorrelated count rate on channel C
    pAB=rAB-rT # uncorrelated count rate on channel AB
    pBC=rBC-rT # uncorrelated count rate on channel BC
    pAC=rAC-rT # uncorrelated count rate on channel AC
    pS=pA+pB+pC # uncorrelated count rate on single channels
    pD=pAB+pBC+pAC # uncorrelated count rate on double coincidence channels
    pT=rT # uncorrelated count rate on triple coincidence channels

    # Calculation of accidental coincidence counting rates
    aAB=(2*(pA*pB+pA*pBC+pB*pAC+pAC*pBC)+(pS+pD-pAB)*(pAB+pT))*t_W*1e-9 # accidental count rate on AB cointing rate
    aBC=(2*(pB*pC+pB*pAC+pC*pAB+pAB*pAC)+(pS+pD-pBC)*(pBC+pT))*t_W*1e-9 # accidental count rate on BC cointing rate
    aAC=(2*(pA*pC+pA*pBC+pC*pAB+pAB*pBC)+(pS+pD-pAC)*(pAC+pT))*t_W*1e-9 # accidental count rate on AC cointing rate
    aD=(2*(pA*pB+pB*pC+pC*pA)+pS*(pD+pT))*t_W*1e-9 # accidental count rate on D cointing rate
    aT=(2*(pA*pBC+pB*pAC+pC*pAB)+(pS+pD)*pT+2*(pBC*pAB+pAC*pBC+pAC*pAB))*t_W*1e-9 # accidental count rate on T cointing rate

    # Corrected count rates
    rAB2=rAB-aAB
    rBC2=rBC-aBC
    rAC2=rAC-aAC
    rD2=rD-aD
    rT2=rT-aT
    
    return rAB2, rBC2, rAC2, rD2, rT2