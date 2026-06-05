# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:50:38 2026

@author: romain.coulon
"""

import readListmode as rlm
import tdcrSubModule as tdsm
import betaGammaSubModule as bg
import spectrum as sp
import configparser
import numpy as np
import tdcrpy as td
import pandas as pd


print("reading...")

config = configparser.ConfigParser()
config.read("config.ini")

rad = config["inputs"]["radionuclide"]
tdcr_active = config["inputs"]["tdcr_active"]
if tdcr_active == 'True':
    tdcr_active = True
    kb = float(config["settings"]["kb"])
    volSource = float(config["settings"]["volSource"])
    n_MCtrial_tdcr = int(config["settings"]["n_MCtrial_tdcr"])
else: tdcr_active = False
bg_active = config["inputs"]["bg_active"]
if bg_active == 'True': bg_active = True
else: bg_active = False


if bg_active:
    lowerBoundGamma = float(config["settings"]["lowerBoundGamma"])
    upperBoundGamma = float(config["settings"]["upperBoundGamma"])
    lowerDelayBound = float(config["settings"]["lowerDelayBound"])
    upperDelayBound = float(config["settings"]["upperDelayBound"])
    gamma_dead_time = float(config["settings"]["gamma_dead_time"])
    
com_ext_deadTime_tdcr = float(config["settings"]["com_ext_deadTime_tdcr"])
coinc_wind_tdcr = float(config["settings"]["coinc_wind_tdcr"])

if bg_active: df_A, df_B, df_C, df_G = rlm.readSeparateCSV(confileFileName="config.ini", bg_active=bg_active)
else: df_A, df_B, df_C = rlm.readSeparateCSV(confileFileName="config.ini", bg_active=bg_active)

print("spectrum analysis...")

sp.energySpectrum(df_A, bins=2**12, label="CH0 beta A")
sp.energySpectrum(df_B, bins=2**12, label="CH0 beta B")
sp.energySpectrum(df_C, bins=2**12, label="CH0 beta C")
if bg_active: 
    sp.energySpectrum(df_G, bins=2**12, label="CH3 NaI gamma", Bounds=[lowerBoundGamma, upperBoundGamma])

    sp.timeDiffSpectrum(df_A, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(A)–γ")
    sp.timeDiffSpectrum(df_B, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(B)–γ")
    sp.timeDiffSpectrum(df_C, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(C)–γ")


    meanDelay = (upperDelayBound + lowerDelayBound) / 2
    bg_window = np.abs(upperDelayBound - lowerDelayBound)

    print(f"\nmean delay time of the gamma channel = {meanDelay} ns")
    print(f"beta-gamma coincidence window = {bg_window} ns")

    print("\nspectrum realignment...")
    df_G['TIMETAG'] = df_G['TIMETAG'] + meanDelay*1e-9
    
    sp.timeDiffSpectrum(df_A, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(A)–γ")
    sp.timeDiffSpectrum(df_A, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(B)–γ")
    sp.timeDiffSpectrum(df_A, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(C)–γ")

print("\ntdcr processing...")

df_beta = tdsm.mergeBeta(df_A, df_B, df_C)
tdcr_count_rates, df_beta_S, df_beta_D, df_beta_T = tdsm.comExtDTprocess(df_beta)
print(f"single event count rate = {tdcr_count_rates['S']} +/- {tdcr_count_rates['u_S']} s-1")
print(f"double event count rate = {tdcr_count_rates['D']} +/- {tdcr_count_rates['u_D']} s-1")
print(f"triple event count rate = {tdcr_count_rates['T']} +/- {tdcr_count_rates['u_T']} s-1")

if tdcr_active:
    print("\ntdcr efficiency calculation...")
    tdcrEff_results = td.TDCRPy.eff(tdcr_count_rates["TDCR"], rad, "1", kb, volSource, N=n_MCtrial_tdcr)
    lightYield, lightYieldABC, eff_beta_S, u_eff_beta_S, eff_beta_D, u_eff_beta_D, eff_beta_T, u_eff_beta_T, _, _, _, _, _, _, _, _ = tdcrEff_results
    print(f"light yield = {lightYield} keV-1")
    print(f"detection efficiency (single) = {eff_beta_S} +/- {u_eff_beta_S}")
    print(f"detection efficiency (double) = {eff_beta_D} +/- {u_eff_beta_D}")
    print(f"detection efficiency (triple) = {eff_beta_T} +/- {u_eff_beta_T}")
    print("\ntdcr activity calculation...")
    activity_beta_S = tdcr_count_rates['S'] / eff_beta_S
    u_activity_beta_S = activity_beta_S * np.sqrt((tdcr_count_rates['u_S'] / tdcr_count_rates['S'])**2 + (u_eff_beta_S / eff_beta_S)**2)
    activity_beta_D = tdcr_count_rates['D'] / eff_beta_D
    u_activity_beta_D = activity_beta_D * np.sqrt((tdcr_count_rates['u_D'] / tdcr_count_rates['D'])**2 + (u_eff_beta_D / eff_beta_D)**2)
    activity_beta_T = tdcr_count_rates['T'] / eff_beta_T
    u_activity_beta_T = activity_beta_T * np.sqrt((tdcr_count_rates['u_T'] / tdcr_count_rates['T'])**2 + (u_eff_beta_T / eff_beta_T)**2)
    print(f"activity (single) = {activity_beta_S} +/- {u_activity_beta_S} Bq")
    print(f"activity (double) = {activity_beta_D} +/- {u_activity_beta_D} Bq")
    print(f"activity (triple) = {activity_beta_T} +/- {u_activity_beta_T} Bq")
    
print("\nbeta-gamma coincidence processing...")
if bg_active:
    print("\ngamma channel processing...")
    data_gamma, df_gamma = bg.gammaExtDTprocess(df_G)
    gamma_count_rate = data_gamma["G"]
    u_gamma_count_rate = data_gamma["u_G"]
    print(f"gamma event count rate = {gamma_count_rate} +/- {u_gamma_count_rate} s-1")
    df_G['CHANNEL'] = 'G'
    
    df_bg_D = bg.mergeBG(df_beta_S, df_beta_D, df_G)
    df_SD = pd.concat([df_beta_S, df_beta_D], ignore_index=True)
    df_bg_T = bg.mergeBG(df_SD, df_beta_T, df_G)
    
    thres_i = 0
    
    data_bg_D, df_coinc_D = bg.bg_process(df_bg_D, thres_i)
    data_bg_T, df_coinc_T = bg.bg_process(df_bg_T, thres_i)
    
    
    # sp.energySpectrum(df_beta_S, bins=2**12, label="beta event (single)")
    # sp.energySpectrum(df_beta_D, bins=2**12, label="beta event (single)")
    # sp.energySpectrum(df_beta_T, bins=2**12, label="beta event (single)")
    
    # bg.bg_process(df_bg_D, thres_i)


