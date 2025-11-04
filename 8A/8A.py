import pandas as pd
import numpy as np
import math
from tabulate import tabulate

#  LOAD DATA 
df = pd.read_csv("mutations.csv")

# --- identify sample column (first column like "Unnamed: 0") ---
sample_col = df.columns[0]
df.rename(columns={sample_col: "Sample"}, inplace=True)

# --- create Class column based on Sample prefix ---
def label_from_sample(name):
    name = str(name).strip().upper()
    if name.startswith("C"):
        return "C"
    elif name.startswith("NC"):
        return "NC"
    else:
        return None

df["Class"] = df["Sample"].apply(label_from_sample)

# --- validate ---
if df["Class"].isnull().any():
    missing = df[df["Class"].isnull()]["Sample"].tolist()[:10]
    raise ValueError(f"Some samples do not start with C or NC. Examples: {missing}")

#  ENTROPY FUNCTION 
def entropy(n_c, n_nc):
    total = n_c + n_nc
    if total == 0:
        return 0.0
    pC = n_c / total
    pNC = n_nc / total
    H = 0
    if pC > 0:
        H -= pC * math.log2(pC)
    if pNC > 0:
        H -= pNC * math.log2(pNC)
    return H

#  ROOT NODE 
n_t = len(df)
n_tC = sum(df["Class"] == "C")
n_tNC = sum(df["Class"] == "NC")
pC_t = n_tC / n_t
pNC_t = n_tNC / n_t
H_t = entropy(n_tC, n_tNC)

print("\nroot node stats")
print(f"n(t)   = {n_t}")
print(f"n(t,C) = {n_tC}")
print(f"n(t,NC)= {n_tNC}")
print(f"pC,t   = {pC_t:.4f}")
print(f"pNC,t  = {pNC_t:.4f}")
print(f"H(t)   = {H_t:.4f}\n")

#  INFORMATION GAIN 
features = [f for f in df.columns if f not in ["Sample", "Class"]]
records = []

for feature in features:
    tL = df[df[feature] == 1]
    tR = df[df[feature] == 0]

    n_tL = len(tL)
    n_tR = len(tR)
    n_tL_C = sum(tL["Class"] == "C")
    n_tL_NC = sum(tL["Class"] == "NC")
    n_tR_C = sum(tR["Class"] == "C")
    n_tR_NC = sum(tR["Class"] == "NC")

    PL = n_tL / n_t
    PR = n_tR / n_t
    H_tL = entropy(n_tL_C, n_tL_NC)
    H_tR = entropy(n_tR_C, n_tR_NC)
    H_s_t = PL * H_tL + PR * H_tR
    gain_s = H_t - H_s_t

    records.append({
        "Feature": feature,
        "n(tL)": n_tL,
        "n(tR)": n_tR,
        "n(tL,C)": n_tL_C,
        "n(tL,NC)": n_tL_NC,
        "n(tR,C)": n_tR_C,
        "n(tR,NC)": n_tR_NC,
        "PL": round(PL, 4),
        "PR": round(PR, 4),
        "H(s,t)": round(H_s_t, 4),
        "H(t)": round(H_t, 4),
        "gain(s)": round(gain_s, 4)
    })

#  TOP 10 TABLE 
sorted_records = sorted(records, key=lambda x: x["gain(s)"], reverse=True)
top10 = sorted_records[:10]

# Assign GEN labels
gene_refs = {f"GEN{i+1}": rec["Feature"] for i, rec in enumerate(top10)}
for i, rec in enumerate(top10):
    rec["Gene"] = f"GEN{i+1}"
    rec.pop("Feature")

#  OUTPUT 
print("gene references")
for k, v in gene_refs.items():
    print(f"{k} = {v}")
print()

headers = ["Gene", "n(tL)", "n(tR)", "n(tL,C)", "n(tL,NC)",
           "n(tR,C)", "n(tR,NC)", "PL", "PR", "H(s,t)", "H(t)", "gain(s)"]
table_data = [[r[h] for h in headers] for r in top10]

print("top 10 by gains")
print(tabulate(table_data, headers=headers, tablefmt="grid"))
