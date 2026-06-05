# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:48:07 2026

@author: romain.coulon
"""

import configparser
import pandas as pd


def readSeparateCSV(confileFileName="config.ini", bg_active=True):
    # ── Load configuration ─────────────────────────────────────────────────────
    config = configparser.ConfigParser()
    config.read(confileFileName)
 
    path_A = config["paths"]["data_path_A"]
    path_B = config["paths"]["data_path_B"]
    path_C = config["paths"]["data_path_C"]
    if bg_active: path_G = config["paths"]["data_path_G"]
 
    # ── Load data (semicolon-separated, keep only TIMETAG & ENERGY) ───────────
    def load(path):
        df = pd.read_csv(path, sep=";", usecols=["TIMETAG", "ENERGY"])
        # TIMETAG: integer ps → float64 seconds (ns precision: 1e-12 * 1e9 = 1e-3 ns steps)
        df["TIMETAG"] = df["TIMETAG"].astype("float64") * 1e-12
        return df
 
    df_A = load(path_A)   # CH0 — beta channel A
    df_B = load(path_B)   # CH1 — beta channel B
    df_C = load(path_C)   # CH2 — beta channel C
    if bg_active: df_G = load(path_G)   # CH3 — gamma (NaI)
 
    print("Loaded data shapes:")
    print(f"  CH0 (A): {df_A.shape}")
    print(f"  CH1 (B): {df_B.shape}")
    print(f"  CH2 (C): {df_C.shape}")
    if bg_active: print(f"  CH3 (G): {df_G.shape}")
 
    if bg_active: return df_A, df_B, df_C, df_G
    else: return df_A, df_B, df_C