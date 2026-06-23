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
import matplotlib.pyplot as plt
import tdcrpy as td
import pandas as pd
import accidentalCoincidenceCorrection as acc


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

    print("\nauto-detecting delay bounds from β(A)–γ time-difference spectrum...")
    lowerDelayBound, upperDelayBound, _ = sp.autoDetectDelayBounds(
        df_A, df_G,
        window=3e-6, bin_width=2e-9,
        energyBoundsG=[lowerBoundGamma, upperBoundGamma],
        label="β(A)–γ"
    )

    sp.timeDiffSpectrum(df_A, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(A)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])
    sp.timeDiffSpectrum(df_B, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(B)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])
    sp.timeDiffSpectrum(df_C, df_G, Bounds=[lowerDelayBound, upperDelayBound], window=3e-6, bin_width=2e-9, label="β(C)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])


    meanDelay = (upperDelayBound + lowerDelayBound) / 2
    bg_window = np.abs(upperDelayBound - lowerDelayBound)

    print(f"\nmean delay time of the gamma channel = {meanDelay} ns")
    print(f"beta-gamma coincidence window = {bg_window} ns")

    print("\nspectrum realignment...")
    df_G['TIMETAG'] = df_G['TIMETAG'] + meanDelay*1e-9
    
    sp.timeDiffSpectrum(df_A, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(A)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])
    sp.timeDiffSpectrum(df_B, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(B)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])
    sp.timeDiffSpectrum(df_C, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(C)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])

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

print("\n**************************************")    
print("beta-gamma coincidence processing...")
print("**************************************")
if bg_active:
    # print("\ngamma channel processing...")
    # data_gamma, df_gamma = bg.gammaExtDTprocess(df_G)
    # gamma_count_rate = data_gamma["G"]
    # u_gamma_count_rate = data_gamma["u_G"]
    # print(f"gamma event count rate = {gamma_count_rate} +/- {u_gamma_count_rate} s-1")
    df_G['CHANNEL'] = 'G'

    # df_beta_S is now CUMULATIVE (every triggered event, multiplicity >= 1),
    # df_beta_D is CUMULATIVE (multiplicity >= 2, doubles AND triples), and
    # df_beta_T holds pure triples only (multiplicity == 3).
    # The "out-of-interest" events for a given beta definition are the
    # triggered events that are NOT used as that beta tag, recovered by
    # removing the chosen beta events (matched on TIMETAG) from df_beta_S:
    #   beta = D : out-of-interest = pure singles            (mult == 1)
    #   beta = T : out-of-interest = pure singles + doubles   (mult == 1 or 2)
    df_out_D = df_beta_S[~df_beta_S["TIMETAG"].isin(df_beta_D["TIMETAG"])].copy()
    df_out_T = df_beta_S[~df_beta_S["TIMETAG"].isin(df_beta_T["TIMETAG"])].copy()

    df_bg_D = bg.mergeBG(df_out_D, df_beta_D, df_G)
    df_bg_T = bg.mergeBG(df_out_T, df_beta_T, df_G)
    
    sp.energySpectrum(df_beta_D, bins=2**12, label="double coincidences")
    sp.energySpectrum(df_beta_T, bins=2**12, label="triple coincidences")
    sp.timeDiffSpectrum(df_beta_D, df_G, Bounds=[-bg_window/2, bg_window/2], window=3e-6, bin_width=2e-9, label="β(D)–γ", energyBoundsG=[lowerBoundGamma, upperBoundGamma])
    
    # --- 1. Pre-Loop Setup for Anticoincidence ---
    # Convert the coincidence window from ns to seconds
    lower_w = -bg_window / 2 * 1e-9
    upper_w =  bg_window / 2 * 1e-9
    
    # Extract the gamma timestamps once (since gamma ROI does not change in the loop)
    # Filter to energy ROI to match the gamma rate used in bg_process
    df_G_ROI = df_G[(df_G["ENERGY"] >= lowerBoundGamma) & (df_G["ENERGY"] <= upperBoundGamma)]
    t_gamma = df_G_ROI["TIMETAG"].values
    duration = df_G["TIMETAG"].iloc[-1] - df_G["TIMETAG"].iloc[0]  # measurement duration in seconds

    # Initialize lists to store anti-coincidence extrapolation data
    anti_eps_D, anti_act_D = [], []
    anti_eps_T, anti_act_T = [], []
    
    threshold_beta_vector = [0, 500, 1000, 2000, 3000]
    
    bg_vec_D, bg_vec_T = [], []
    u_bg_vec_D, u_bg_vec_T = [], []
    b_vec_D, b_vec_T = [], []
    u_b_vec_D, u_b_vec_T = [], []
    g_vec_D, g_vec_T = [], []
    u_g_vec_D, u_g_vec_T = [], []

    for thres_i in threshold_beta_vector:
        data_bg_D, df_coinc_D = bg.bg_process(df_bg_D, thres_i)
        bg_vec_D.append(data_bg_D["BG_corr"])
        u_bg_vec_D.append(data_bg_D["u_BG_corr"])
        b_vec_D.append(data_bg_D["B"])
        u_b_vec_D.append(data_bg_D["u_B"])
        g_vec_D.append(data_bg_D["G"])
        u_g_vec_D.append(data_bg_D["u_G"])

        data_bg_T, df_coinc_T = bg.bg_process(df_bg_T, thres_i)
        bg_vec_T.append(data_bg_T["BG_corr"])
        u_bg_vec_T.append(data_bg_T["u_BG_corr"])
        b_vec_T.append(data_bg_T["B"])
        u_b_vec_T.append(data_bg_T["u_B"])
        g_vec_T.append(data_bg_T["G"])
        u_g_vec_T.append(data_bg_T["u_G"])

        print(f"\nthreshold = {thres_i}")
        print(f"  beta  count rate        (D) = {data_bg_D['B']:.4f} +/- {data_bg_D['u_B']:.4f} s-1")
        print(f"  gamma count rate        (D) = {data_bg_D['G']:.4f} +/- {data_bg_D['u_G']:.4f} s-1")
        print(f"  beta-gamma raw          (D) = {data_bg_D['BG']:.4f} +/- {data_bg_D['u_BG']:.4f} s-1")
        print(f"  beta-gamma corrected    (D) = {data_bg_D['BG_corr']:.4f} +/- {data_bg_D['u_BG_corr']:.4f} s-1")
        print(f"    acc. R_f(gamma->beta) (D) = {data_bg_D['R_acc_gamma_to_beta']:.6f} s-1")
        print(f"    acc. R_f(beta->gamma) (D) = {data_bg_D['R_acc_beta_to_gamma']:.6f} s-1")
        print(f"  beta  count rate        (T) = {data_bg_T['B']:.4f} +/- {data_bg_T['u_B']:.4f} s-1")
        print(f"  gamma count rate        (T) = {data_bg_T['G']:.4f} +/- {data_bg_T['u_G']:.4f} s-1")
        print(f"  beta-gamma raw          (T) = {data_bg_T['BG']:.4f} +/- {data_bg_T['u_BG']:.4f} s-1")
        print(f"  beta-gamma corrected    (T) = {data_bg_T['BG_corr']:.4f} +/- {data_bg_T['u_BG_corr']:.4f} s-1")
        print(f"    acc. R_f(gamma->beta) (T) = {data_bg_T['R_acc_gamma_to_beta']:.6f} s-1")
        print(f"    acc. R_f(beta->gamma) (T) = {data_bg_T['R_acc_beta_to_gamma']:.6f} s-1")
        
        # ---------------------------------------------------------
        # B. NEW ANTICOINCIDENCE LOGIC
        # ---------------------------------------------------------
        # Dynamically filter beta events above the current threshold
        t_beta_D = df_beta_D[df_beta_D['ENERGY'] > thres_i]["TIMETAG"].values
        t_beta_T = df_beta_T[df_beta_T['ENERGY'] > thres_i]["TIMETAG"].values
        
        # Evaluate Double Anticoincidences
        N_g_D, N_raw_D, N_acc_D = acc.evaluate_listmode_coincidences(
            t_beta=t_beta_D, t_gamma=t_gamma, 
            window_lower=lower_w, window_upper=upper_w, shift_time=100e-6
        )
        anti_res_D = acc.process_anticoincidence_activity(len(t_beta_D), N_g_D, N_raw_D, N_acc_D, duration)
        anti_eps_D.append(anti_res_D["efficiency_beta"])
        anti_act_D.append(anti_res_D["Activity_N0"])
    
        # Evaluate Triple Anticoincidences
        N_g_T, N_raw_T, N_acc_T = acc.evaluate_listmode_coincidences(
            t_beta=t_beta_T, t_gamma=t_gamma, 
            window_lower=lower_w, window_upper=upper_w, shift_time=100e-6
        )
        anti_res_T = acc.process_anticoincidence_activity(len(t_beta_T), N_g_T, N_raw_T, N_acc_T, duration)
        anti_eps_T.append(anti_res_T["efficiency_beta"])
        anti_act_T.append(anti_res_T["Activity_N0"])
    
    print("**********************************")
    print("activity extrapolation analysis...")
    print("**********************************")
    # ── Activity vs. (1 - beta efficiency) extrapolation ──────────────────────
    # Beta efficiency seen by the gamma-tagged sample: eff_beta = BG / G
    # Activity estimate (Premillieu / 4pi-beta-gamma extrapolation method):
    #     A = B * G / BG
    # A is plotted against the beta-channel inefficiency (1 - eff_beta) and
    # linearly extrapolated to inefficiency = 0 (i.e. eff_beta = 1) to obtain
    # the dead-time/efficiency-independent activity.

    def activity_extrapolation(b_vec, g_vec, bg_vec, u_b_vec, u_g_vec, u_bg_vec, label=""):
        b_vec    = np.array(b_vec,    dtype=float)
        g_vec    = np.array(g_vec,    dtype=float)
        bg_vec   = np.array(bg_vec,   dtype=float)
        u_b_vec  = np.array(u_b_vec,  dtype=float)
        u_g_vec  = np.array(u_g_vec,  dtype=float)
        u_bg_vec = np.array(u_bg_vec, dtype=float)

        eff_beta   = bg_vec / g_vec
        # eff_beta = (bg_vec / g_vec) * (mean_lt_gamma / mean_lt_beta)
        u_eff_beta = eff_beta * np.sqrt((u_bg_vec / bg_vec)**2 + (u_g_vec / g_vec)**2)

        # ── Two abscissa choices ───────────────────────────────────────────────
        # x1 : (1 - ε_β)          — classical Campion inefficiency
        # x2 : (1 - ε_β) / ε_β   — ratio form (linear extrapolation to 0
        #                            also gives A at ε_β = 1)
        x1   = 1.0 - eff_beta
        u_x1 = u_eff_beta

        # Propagation for x2 = (1 - ε) / ε :  dx2/dε = -1/ε²
        x2   = x1 / eff_beta
        u_x2 = u_eff_beta / eff_beta**2   # |dx2/dε| · u_ε

        activity   = b_vec * g_vec / bg_vec
        u_activity = activity * np.sqrt(
            (u_b_vec / b_vec)**2 + (u_g_vec / g_vec)**2 + (u_bg_vec / bg_vec)**2
        )

        # ── Helper: fit + plot for one (x, fit_order) configuration ──────────
        def _fit_and_plot(x, u_x, x_label, poly_order, config_label):
            coeffs, cov = np.polyfit(x, activity, poly_order, cov=True)
            # intercept is always the last coefficient (value at x = 0)
            intercept   = coeffs[-1]
            u_intercept = np.sqrt(cov[-1, -1])

            order_str = "linear" if poly_order == 1 else f"poly{poly_order}"
            print(f"\n  [{config_label}] {order_str} fit on {x_label}:")
            for xi, a, ua in zip(x, activity, u_activity):
                print(f"    {x_label} = {xi:.4f}  ->  A = {a:.4f} +/- {ua:.4f} Bq")
            print(f"    extrapolated activity (ε_β → 1) = {intercept:.4f} +/- {u_intercept:.4f} Bq")

            x_fit  = np.linspace(0, max(x.max(), 0.01) * 1.1, 200)
            y_fit  = np.polyval(coeffs, x_fit)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(x, activity, yerr=u_activity, xerr=u_x,
                        fmt="o", capsize=3, label="data")
            ax.plot(x_fit, y_fit, "--",
                    label=f"{order_str}: A₀ = {intercept:.4f} ± {u_intercept:.4f} Bq")
            ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Activity (Bq)")
            ax.set_title(f"Activity extrapolation — {label} [{config_label}]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            return intercept, u_intercept

        # ── Run all 4 configurations ───────────────────────────────────────────
        print(f"\nActivity extrapolation ({label}):")

        results = {}
        for x, u_x, x_lbl, x_key in [
            (x1, u_x1, r"$1 - \varepsilon_\beta$",                    "x1"),
            (x2, u_x2, r"$(1 - \varepsilon_\beta)/\varepsilon_\beta$", "x2"),
        ]:
            for order, o_key in [(1, "lin"), (2, "poly2")]:
                key = f"{x_key}_{o_key}"
                cfg_label = f"{'linear' if order == 1 else 'poly2'}, {x_lbl}"
                a0, u_a0 = _fit_and_plot(x, u_x, x_lbl, order, cfg_label)
                results[key] = (a0, u_a0)

        # Return the classical linear result (x1, order=1) as primary output
        intercept, u_intercept = results["x1_lin"]
        
        print(f"  eff_beta range: {eff_beta.min():.4f} – {eff_beta.max():.4f}")
        print(f"  activity range: {activity.min():.4f} – {activity.max():.4f}")
        
        
        return results

    results_bg_D = \
        activity_extrapolation(b_vec_D, g_vec_D, bg_vec_D, u_b_vec_D, u_g_vec_D, u_bg_vec_D, label="double coincidences")

    results_bg_T = \
        activity_extrapolation(b_vec_T, g_vec_T, bg_vec_T, u_b_vec_T, u_g_vec_T, u_bg_vec_T, label="triple coincidences")

    print("\nRESUTS FROM BETA-GAMMA COINCIDENCE")
    print(f"\nActivity (double, lin fit, 1 - epsilon_beta) = {results_bg_D['x1_lin'][0]} +/- {results_bg_D['x1_lin'][1]} Bq")
    print(f"Activity (double, poly fit, 1 - epsilon_beta) = {results_bg_D['x1_poly2'][0]} +/- {results_bg_D['x1_poly2'][1]} Bq")
    print(f"Activity (double, lin fit, (1 - epsilon_beta)/epsilon_beta) = {results_bg_D['x2_lin'][0]} +/- {results_bg_D['x2_lin'][1]} Bq")
    print(f"Activity (double, poly fit, (1 - epsilon_beta)/epsilon_beta) = {results_bg_D['x2_poly2'][0]} +/- {results_bg_D['x2_poly2'][1]} Bq")
    
    print(f"Activity (triple, lin fit, 1 - epsilon_beta) = {results_bg_T['x1_lin'][0]} +/- {results_bg_T['x1_lin'][1]} Bq")
    print(f"Activity (triple, poly fit, 1 - epsilon_beta) = {results_bg_T['x1_poly2'][0]} +/- {results_bg_T['x1_poly2'][1]} Bq")
    print(f"Activity (triple, lin fit, (1 - epsilon_beta)/epsilon_beta) = {results_bg_T['x2_lin'][0]} +/- {results_bg_T['x2_lin'][1]} Bq")
    print(f"Activity (triple, poly fit, (1 - epsilon_beta)/epsilon_beta) = {results_bg_T['x2_poly2'][0]} +/- {results_bg_T['x2_poly2'][1]} Bq")
    
    # --- 2. Post-Loop Extrapolation ---

    # Calculate the extrapolation parameter: x2 = (1 - epsilon) / epsilon
    x2_anti_D = (1 - np.array(anti_eps_D)) / np.array(anti_eps_D)
    x2_anti_T = (1 - np.array(anti_eps_T)) / np.array(anti_eps_T)
    
    # Convert activity arrays for fitting
    y_anti_D = np.array(anti_act_D)
    y_anti_T = np.array(anti_act_T)
    
    # Perform linear and polynomial fits (intercept is the last element, [-1])
    fit_anti_lin_D = np.polyfit(x2_anti_D, y_anti_D, 1)
    fit_anti_poly_D = np.polyfit(x2_anti_D, y_anti_D, 2)
    
    fit_anti_lin_T = np.polyfit(x2_anti_T, y_anti_T, 1)
    fit_anti_poly_T = np.polyfit(x2_anti_T, y_anti_T, 2)
    
    # (Your existing prints here...)
    print("\nRESULTS FROM BETA-GAMMA COINCIDENCE")
    print(f"\nActivity (double, lin fit, 1 - epsilon_beta) = {results_bg_D['x1_lin'][0]} +/- {results_bg_D['x1_lin'][1]} Bq")
    # ... 
    
    # New Anticoincidence prints
    print("\n**************************************")
    print("RESULTS FROM BETA-GAMMA ANTI-COINCIDENCE (SHIFT METHOD)")
    print("**************************************")
    
    # For Anticoincidence, the parameter of choice is almost always (1-eps)/eps.
    print(f"\nActivity (double, lin fit, (1 - epsilon_beta)/epsilon_beta) = {fit_anti_lin_D[-1]:.2f} Bq")
    print(f"Activity (double, poly fit, (1 - epsilon_beta)/epsilon_beta) = {fit_anti_poly_D[-1]:.2f} Bq")
    
    print(f"\nActivity (triple, lin fit, (1 - epsilon_beta)/epsilon_beta) = {fit_anti_lin_T[-1]:.2f} Bq")
    print(f"Activity (triple, poly fit, (1 - epsilon_beta)/epsilon_beta) = {fit_anti_poly_T[-1]:.2f} Bq")