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


def readSeparateCSV_bkg(confileFileName="config.ini"):
    """
    Load background measurement files defined by data_path_*_bck keys in config.ini.

    Returns
    -------
    df_A_bck, df_B_bck, df_C_bck, df_G_bck : DataFrames with TIMETAG (s) and ENERGY
    """
    config = configparser.ConfigParser()
    config.read(confileFileName)

    def load(path):
        df = pd.read_csv(path, sep=";", usecols=["TIMETAG", "ENERGY"])
        df["TIMETAG"] = df["TIMETAG"].astype("float64") * 1e-12
        return df

    df_A_bck = load(config["paths"]["data_path_A_bck"])
    df_B_bck = load(config["paths"]["data_path_B_bck"])
    df_C_bck = load(config["paths"]["data_path_C_bck"])
    df_G_bck = load(config["paths"]["data_path_G_bck"])

    print("Loaded background data shapes:")
    print(f"  CH0 (A_bck): {df_A_bck.shape}")
    print(f"  CH1 (B_bck): {df_B_bck.shape}")
    print(f"  CH2 (C_bck): {df_C_bck.shape}")
    print(f"  CH3 (G_bck): {df_G_bck.shape}")

    return df_A_bck, df_B_bck, df_C_bck, df_G_bck