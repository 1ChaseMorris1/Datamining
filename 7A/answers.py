import pandas as pd

# === step 1: load data ===
df = pd.read_csv("mutations.csv")

if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "sample"})

df["label"] = ["C" if str(x).startswith("C") else "NC" for x in df["sample"]]
features = [c for c in df.columns if c not in ("sample", "label")]

# root node summary
n_t = len(df)
n_t_C = (df["label"] == "C").sum()
n_t_NC = (df["label"] == "NC").sum()
print("root node summary:")
print(f"n(t)={n_t}  n(t,c)={n_t_C}  n(t,nc)={n_t_NC}")
print(f"pc,t={n_t_C/n_t:.4f}  pnc,t={n_t_NC/n_t:.4f}\n")

# === step 2: compute f(s,t) ===
records = []
for feature in features:
    left = df[df[feature] == 0]
    right = df[df[feature] == 1]
    n_tL, n_tR = len(left), len(right)
    if n_tL == 0 or n_tR == 0:
        continue

    n_tL_C = (left["label"] == "C").sum()
    n_tL_NC = (left["label"] == "NC").sum()
    n_tR_C = (right["label"] == "C").sum()
    n_tR_NC = (right["label"] == "NC").sum()

    PL, PR = n_tL / n_t, n_tR / n_t
    twoPLPR = 2 * PL * PR
    P_C_tL, P_NC_tL = n_tL_C / n_tL, n_tL_NC / n_tL
    P_C_tR, P_NC_tR = n_tR_C / n_tR, n_tR_NC / n_tR
    Q = abs(P_C_tL - P_C_tR) + abs(P_NC_tL - P_NC_tR)
    F_value = twoPLPR * Q

    records.append({
        "mutation": feature,
        "n(tl)": n_tL, "n(tr)": n_tR,
        "n(tl,c)": n_tL_C, "n(tl,nc)": n_tL_NC,
        "pl": PL, "pr": PR,
        "p(c|tl)": P_C_tL, "p(nc|tl)": P_NC_tL,
        "p(c|tr)": P_C_tR, "p(nc|tr)": P_NC_tR,
        "2plpr": twoPLPR, "q": Q, "f(s,t)": F_value
    })

results = pd.DataFrame(records).sort_values("f(s,t)", ascending=False).head(10).reset_index(drop=True)

# === step 3: abbreviations ===
abbrevs = {}
for i, mut in enumerate(results["mutation"], start=1):
    abbrevs[f"mut{i}"] = mut
results["abbr"] = [f"mut{i}" for i in range(1, len(results) + 1)]

# === step 4: compact sql-style print ===
cols = ["abbr", "n(tl)", "n(tr)", "n(tl,c)", "n(tl,nc)",
        "pl", "pr", "p(c|tl)", "p(nc|tl)", "p(c|tr)",
        "p(nc|tr)", "2plpr", "q", "f(s,t)"]

# reduce widths for compact layout
widths = {c: 7 for c in cols}
widths["abbr"] = 6
widths["mutation"] = 10

header = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"
line = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"

print("top 10 features by f(s,t)\n")
print(line)
print(header)
print(line)
for _, row in results.iterrows():
    print("| " + " | ".join(
        f"{(row[c] if not isinstance(row[c], float) else f'{row[c]:.4f}'):<{widths[c]}}"
        for c in cols) + " |")
print(line)

# === step 5: legend ===
print("\nlegend:")
for abbr, full_name in abbrevs.items():
    print(f"  {abbr} = {full_name}")
