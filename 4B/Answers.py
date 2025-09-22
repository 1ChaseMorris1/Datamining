import pandas as pd

# read in file.
file_path = "mutations.csv" 
df = pd.read_csv(file_path)

# rename the first columb or list of names to "sample."
df = df.rename(columns={"Unnamed: 0": "Sample"})

# Simplifies sample, if the name starts with C -> C if not -> NC.
df["Sample"] = df["Sample"].str.startswith("C").map({True: "C", False: "NC"})

# create a table of all the feature data excluding the sample columb. 
feature = df.drop(columns=["Sample"])

tpFpList = []

for feature in feature.columns:

    # tp = C and 1 so sum all in C.
    tp = df[df["Sample"] == "C"][feature].sum()

    # fp = NC and 1 so sum all in NC.
    fp = df[df["Sample"] == "NC"][feature].sum()
    
    #add the tp - fp to a list.
    tpFpList.append((feature, tp - fp))

# sort the information by the second data in the touple and reverse it.
tpFpList.sort(key=lambda x: x[1], reverse=True)

# print the top 10 in list
print("\n\t\t\tTop 10 Features:")
for item in tpFpList[:10]: 
    print(item)

# print the most helpful feature or just the first item in sample
F = tpFpList[0][0]
print("\n\t\t\tMost Helpful Feature")
print(F, "\n")

# TP is where sample = C and feature = 1. 
TP = ((df["Sample"] == "C") & (df[F] == 1)).sum()

# FP is where sample = NC and feature = 1.
FP = ((df["Sample"] == "NC") & (df[F] == 1)).sum()

# TN is where sample = NC and feature = 0.
TN = ((df["Sample"] == "NC") & (df[F] == 0)).sum()

# FN is where sample = C and feature = 0. 
FN = ((df["Sample"] == "C") & (df[F] == 0)).sum()

# print top row
print("\tCancer\tNo Cancer")

#print second row
print("Actual C:", TP, "\t  ", FN)
#print third row
print("Actual N:", FP, "\t  ", TN)

# group the samples
A = df[df[F] == 1]["Sample"].tolist()
B = df[df[F] == 0]["Sample"].tolist()

# print out groups by feature
print("\nGroups by Feature:")
print(f"Group-A (Has {F}): {len(A)} samples")
print(f"Group-B (No {F}): {len(B)} samples")


