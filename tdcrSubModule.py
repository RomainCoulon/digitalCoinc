# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:32:29 2026

@author: romain.coulon
"""
import pandas as pd
import accidentalCoincidenceCorrection as acc
import configparser
import numpy as np


def mergeBeta(df_A, df_B, df_C):
    """
    Merge beta channels A, B, C into a single time-sorted DataFrame.
 
    Parameters
    ----------
    df_A, df_B, df_C : DataFrames with TIMETAG (s, float64) and ENERGY columns
 
    Returns
    -------
    df_beta : merged and time-sorted DataFrame with an added 'CHANNEL' column
    """
    df_A = df_A.copy(); df_A["CHANNEL"] = "A"
    df_B = df_B.copy(); df_B["CHANNEL"] = "B"
    df_C = df_C.copy(); df_C["CHANNEL"] = "C"
 
    df_beta = pd.concat([df_A, df_B, df_C], ignore_index=True)
    df_beta.sort_values("TIMETAG", inplace=True)
    df_beta.reset_index(drop=True, inplace=True)
 
    print(f"Merged beta shape: {df_beta.shape}")
    return df_beta
    

def comExtDTprocess(df_beta, confileFileName="config.ini"):
    """
    TDCR dead-time processing with extended dead time model.
 
    Parameters
    ----------
    df_beta : time-sorted DataFrame with columns TIMETAG (s), ENERGY, CHANNEL (A/B/C)
    confileFileName : path to config.ini
 
    Returns
    -------
    data_tdcr : dict of count rates, uncertainties, TDCR parameter, live/real time
    """
    import accidentalCoincidenceCorrection as acc
 
    config = configparser.ConfigParser()
    config.read(confileFileName)
    ExtDT      = config["settings"].getfloat("com_ext_deadTime_tdcr") * 1e-9   # ns → s
    ResolTime  = config["settings"].getfloat("coinc_wind_tdcr")        * 1e-9   # ns → s
    thres      = config["settings"].getfloat("threshold_tdcr")
    n_run_tdcr = config["settings"].getint("n_run_tdcr")
 
    # ── Counter initialisation ─────────────────────────────────────────────────
    triplet = [False, False, False]
    Acount = Bcount = Ccount = 0
    ABcount = BCcount = ACcount = 0
    Scount = Dcount = Tcount = 0
 
    # ── Time parameter initialisation ──────────────────────────────────────────
    trigger_time = 0.0   # time of the previous trigger
    comdtW       = 0.0   # projected dead time window
    realTime     = 0.0   # current real time
    liveTime     = 0.0   # accumulated live time
    count_pulse  = 0     # global event counter
    w            = 1     # intermediate result index
 
    # ── Intermediate result lists ──────────────────────────────────────────────
    Arate  = []; Brate  = []; Crate  = []
    ABrate = []; BCrate = []; ACrate = []
    Srate  = []; Drate  = []; Trate  = []
    lt     = []   # live times per run
    single_events = []   # (trigger_time, energy) for single events
    double_events = []   # (trigger_time, energy_sum) for double coincidences
    triple_events = []   # (trigger_time, energy_sum) for triple coincidences
 
    def resetTriplet():
        return [False, False, False]
 
    triplet_energy = [0.0, 0.0, 0.0]   # energy recorded per channel in the current window
 
    def resetTripletEnergy():
        return [0.0, 0.0, 0.0]
 
    # Duration of each intermediate run (s)
    t_mes = df_beta["TIMETAG"].iloc[-1] / n_run_tdcr
 
    # ── Main loop ──────────────────────────────────────────────────────────────
    for row in df_beta.itertuples(index=False):
        t_evt   = row.TIMETAG   # arrival time of this event (s)
        charge  = row.ENERGY
        channel = row.CHANNEL
 
        realTime = t_evt
 
        # ── (1) EVENT ARRIVES WHEN SYSTEM IS IDLE ─────────────────────────────
        if charge >= thres and t_evt > trigger_time + comdtW:
 
            # (1.1) Record the previous triplet buffer into counters
            if sum(triplet) >= 1:
                Scount += 1
                energy_sum = sum(triplet_energy)

                # df_single ("S"): EVERY triggering event, whatever its
                # multiplicity (single, double or triple-fold). This follows
                # the standard TDCR convention where nS is the "at least one
                # channel fired" rate, i.e. nS ⊇ nD ⊇ nT. It is what allows
                # the out-of-interest sets used downstream (see
                # outOfInterest()) to be obtained by simple set differences.
                single_events.append((trigger_time, energy_sum))

                if triplet == [True, False, False]: Acount += 1
                if triplet == [False, True, False]: Bcount += 1
                if triplet == [False, False, True]: Ccount += 1

                if sum(triplet) >= 2:
                    Dcount += 1

                    # df_double ("D"): every event with at least two channels
                    # fired, i.e. pure doubles AND triples (a triple
                    # coincidence is, by construction, also a double
                    # coincidence on every pair of channels).
                    double_events.append((trigger_time, energy_sum))

                    if sum(triplet) == 2:
                        # Pure double coincidence (exactly two channels fired)
                        if triplet == [True, True, False]:  ABcount += 1
                        if triplet == [False, True, True]:  BCcount += 1
                        if triplet == [True, False, True]:  ACcount += 1
                    else:
                        # Triple coincidence (all three channels fired)
                        # A triple is simultaneously AB, BC, and AC
                        ABcount += 1; BCcount += 1; ACcount += 1
                        Tcount += 1

                        # df_triple ("T"): pure triples only (already the
                        # maximal multiplicity, so "exactly three" = "at
                        # least three").
                        triple_events.append((trigger_time, energy_sum))
 
            # (1.1 opt) Intermediate result at end of each run
            if realTime > t_mes * w:
                print(f"run #{w}  {int(realTime / 60)} min processed.")
                if liveTime > 0:
                    Arate.append(Acount  / liveTime)
                    Brate.append(Bcount  / liveTime)
                    Crate.append(Ccount  / liveTime)
                    ABrate.append(ABcount / liveTime)
                    BCrate.append(BCcount / liveTime)
                    ACrate.append(ACcount / liveTime)
                    Srate.append(Scount  / liveTime)
                    Drate.append(Dcount  / liveTime)
                    Trate.append(Tcount  / liveTime)
                    lt.append(liveTime)
                # Reinitialise counters
                Acount = Bcount = Ccount = 0
                ABcount = BCcount = ACcount = 0
                Scount = Dcount = Tcount = 0
                liveTime = 0.0
                w += 1
 
            # (1.2) Update live time
            liveTime += t_evt - (trigger_time + comdtW)
 
            # (1.3) Reinitialise buffers for new event
            triplet        = resetTriplet()
            triplet_energy = resetTripletEnergy()
            trigger_time   = t_evt
            if channel == "A": triplet[0] = True; triplet_energy[0] = charge
            if channel == "B": triplet[1] = True; triplet_energy[1] = charge
            if channel == "C": triplet[2] = True; triplet_energy[2] = charge
            comdtW      = ExtDT
            count_pulse = 1
 
        # ── (2) EVENT ARRIVES WHEN SYSTEM IS BUSY ─────────────────────────────
        elif charge >= thres:
            count_pulse += 1
            # (2.1) Within the coincidence resolving time → add to triplet
            if t_evt < trigger_time + ResolTime:
                if channel == "A": triplet[0] = True; triplet_energy[0] += charge
                if channel == "B": triplet[1] = True; triplet_energy[1] += charge
                if channel == "C": triplet[2] = True; triplet_energy[2] += charge
                comdtW += t_evt - trigger_time   # extend paralysing dead time
            # (2.2) Outside resolving time → extend dead time from hidden event
            else:
                comdtW = t_evt - trigger_time + ExtDT
 
    # ── Flush the last (incomplete) run ───────────────────────────────────────
    if liveTime > 0:
        print(f"run #{w}  {int(realTime / 60)} min processed. [last]")
        Arate.append(Acount  / liveTime)
        Brate.append(Bcount  / liveTime)
        Crate.append(Ccount  / liveTime)
        ABrate.append(ABcount / liveTime)
        BCrate.append(BCcount / liveTime)
        ACrate.append(ACcount / liveTime)
        Srate.append(Scount  / liveTime)
        Drate.append(Dcount  / liveTime)
        Trate.append(Tcount  / liveTime)
        lt.append(liveTime)
 
    n_actual = len(Arate)   # may differ from n_run_tdcr by ±1
 
    # ── Accidental coincidence correction ──────────────────────────────────────
    ab = np.empty(n_actual); bc = np.empty(n_actual); ac = np.empty(n_actual)
    d  = np.empty(n_actual); tr = np.empty(n_actual)
    s  = np.empty(n_actual)   # corrected singles (filled below)
 
    for i in range(n_actual):
        ab[i], bc[i], ac[i], d[i], tr[i] = acc.accidentalCoincCorr(
            Arate[i], Brate[i], Crate[i],
            ABrate[i], BCrate[i], ACrate[i],
            Drate[i], Trate[i],
            ResolTime * 1e9
        )
        s[i] = Srate[i]   # ← fix: was Arate[i]+Brate[i]+Crate[i]
 
    # ── Means and standard deviations ─────────────────────────────────────────
    data_tdcr = {
        "thres_lbs":          thres,
        "A":  np.mean(Arate),  "u_A":  np.std(Arate),
        "B":  np.mean(Brate),  "u_B":  np.std(Brate),
        "C":  np.mean(Crate),  "u_C":  np.std(Crate),
        "AB": np.mean(ab),     "u_AB": np.std(ab),
        "BC": np.mean(bc),     "u_BC": np.std(bc),
        "AC": np.mean(ac),     "u_AC": np.std(ac),
        "S":  np.mean(s),      "u_S":  np.std(s),
        "D":  np.mean(d),      "u_D":  np.std(d),
        "T":  np.mean(tr),     "u_T":  np.std(tr),
        "Check_sum":           bool(np.mean(ab) + np.mean(bc) + np.mean(ac) - 2*np.mean(tr) == np.mean(d)),
        "TDCR":                np.mean(tr / d),
        "u_TDCR":              np.std(tr / d),
        "Dead_time_percent":   round(100 - 100 * sum(lt) / realTime, 1),
        "Live_time_s":         lt,
        "Real_time_s":         realTime,
    }
 
    # ── Build filtered DataFrames for double and triple events ───────────────
    df_single = pd.DataFrame(single_events, columns=["TIMETAG", "ENERGY"])
    df_single["CHANNEL"] = "S"
 
    df_double = pd.DataFrame(double_events, columns=["TIMETAG", "ENERGY"])
    df_double["CHANNEL"] = "D"
 
    df_triple = pd.DataFrame(triple_events, columns=["TIMETAG", "ENERGY"])
    df_triple["CHANNEL"] = "T"
 
    print(f"Single events:             {len(df_single)}")
    print(f"Double coincidence events: {len(df_double)}")
    print(f"Triple coincidence events: {len(df_triple)}")
 
    return data_tdcr, df_single, df_double, df_triple