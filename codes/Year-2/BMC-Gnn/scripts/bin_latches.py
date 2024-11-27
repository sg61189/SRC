#!/usr/bin/env python3
"""Produce 5 bins of circuits based on total gate counts."""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/gate_counts.csv")
counts, boundaries, _ = plt.hist(df["M"], bins=5)

df_bin1 = df[(df["M"] >= boundaries[0]) & (df["M"] < boundaries[1])]
df_bin2 = df[(df["M"] >= boundaries[1]) & (df["M"] < boundaries[2])]
df_bin3 = df[(df["M"] >= boundaries[2]) & (df["M"] < boundaries[3])]
df_bin4 = df[(df["M"] >= boundaries[3]) & (df["M"] < boundaries[4])]
df_bin5 = df[(df["M"] >= boundaries[4]) & (df["M"] < boundaries[5])]

print(df_bin1)
print(df_bin2)
print(df_bin3)
print(df_bin4)
print(df_bin5)

plt.show()
