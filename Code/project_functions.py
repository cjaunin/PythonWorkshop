#!/usr/bin/env python3
########################################################################################################################
# EMF-QARM - Python Workshop - Functions
# Authors: Maxime Borel, Coralie Jaunin
# Creation Date: February 27, 2023
# Revised on: February 27, 2023
########################################################################################################################
# winsorize

def winsorize(df, alpha, up=True, down=True):
    if up:
        ub = df.quantile(1-alpha, interpolation='higher', axis=0)
    else:
        ub = df.max(axis=0)
    if down:
        lb = df.quantile(alpha, interpolation='lower', axis=0)
    else:
        lb = df.min(axis=0)
    df_win = df.clip(lower=lb, upper=ub, axis=1)
    return df_win