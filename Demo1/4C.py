import warnings                                                                             # Import the warnings module to suppress specific runtime warnings
warnings.filterwarnings("ignore", message="Signature .* for <class 'numpy.longdouble'>.*")  # Ignore a known NumPy longdouble warning

import pandas as pd                                                            # Import pandas for data handling

df = pd.read_csv("mutations.csv")                                              # Read the CSV file containing mutation data.
df = df.rename(columns={"Unnamed: 0": "Sample"})                               # Rename the first unnamed column to "Sample"
groups = df.copy()                                                             # Keep a copy of the original dataset for later grouping
df["Sample"] = df["Sample"].str.startswith("C").map({True: "C", False: "NC"})  # Map samples starting with 'C' to 'C' (Cancer) else 'NC'

tpFpList = []                                                      # Initialize a list to hold (feature, TP-FP) tuples
for f in df.drop(columns=["Sample"]).columns:                      # Loop through each feature column, ignoring the 'Sample' column
    tp = df[df["Sample"] == "C"][f].sum()                          # Count true positives: samples labeled C with feature=1
    fp = df[df["Sample"] == "NC"][f].sum()                         # Count false positives: samples labeled NC with feature=1
    tpFpList.append((f, tp - fp))                                  # Append (feature, TP-FP value) to list
tpFpList.sort(key=lambda x: x[1], reverse=True)                    # Sort features by TP-FP descending to find most discriminative

print("\n\t\t\tTop 10 Features:")                                  # Header for top 10 features list
for item in tpFpList[:10]:                                         # Loop over the first 10 highest TP-FP features
    print(item)                                                    # Print feature name and its TP-FP score

F = tpFpList[0][0]                                                 # Select the feature with highest TP-FP as root (F)
print("\n\t\t\tMost Helpful Feature")                              # Header for best feature
print(F, "\n")                                                     # Print the chosen root feature

# Confusion matrix for F
TP = ((df["Sample"] == "C") & (df[F] == 1)).sum()                  # Count: C samples with F=1 (true positives)
FP = ((df["Sample"] == "NC") & (df[F] == 1)).sum()                 # Count: NC samples with F=1 (false positives)
TN = ((df["Sample"] == "NC") & (df[F] == 0)).sum()                 # Count: NC samples with F=0 (true negatives)
FN = ((df["Sample"] == "C") & (df[F] == 0)).sum()                  # Count: C samples with F=0 (false negatives)

print("\tCancer\tNo Cancer")                                       # Header row for confusion matrix
print("Actual C:", TP, "\t  ", FN)                                 # Print counts for actual Cancer (TP and FN)
print("Actual N:", FP, "\t  ", TN)                                 # Print counts for actual No Cancer (FP and TN)

# Split
group_A = df[df[F] == 1].copy()                                    # Create subgroup A: samples where F=1
group_B = df[df[F] == 0].copy()                                    # Create subgroup B: samples where F=0

tpFpListA = []                                                     # Initialize list for (feature, TP-FP) inside group A
for f in group_A.drop(columns=["Sample"]).columns:                 # Loop through features within group A
    tp = group_A[group_A["Sample"] == "C"][f].sum()                # Count C with feature=1 in group A
    fp = group_A[group_A["Sample"] == "NC"][f].sum()               # Count NC with feature=1 in group A
    tpFpListA.append((f, tp - fp))                                 # Append (feature, TP-FP) for group A
tpFpListA.sort(key=lambda x: x[1], reverse=True)                   # Sort features by TP-FP for group A

# group A
print("\n\t\t\tTop 10 Features (Group A):")                        # Header for group A top features
for item in tpFpListA[:10]:                                        # Loop through top 10 features in group A
    print(item)                                                    # Print feature name and TP-FP value
A = tpFpListA[0][0]                                                # Select best feature for group A

TPa = ((group_A["Sample"] == "C") & (group_A[A] == 1)).sum()       # True positives in group A using feature A
FPa = ((group_A["Sample"] == "NC") & (group_A[A] == 1)).sum()      # False positives in group A using feature A
TNa = ((group_A["Sample"] == "NC") & (group_A[A] == 0)).sum()      # True negatives in group A using feature A
FNa = ((group_A["Sample"] == "C") & (group_A[A] == 0)).sum()       # False negatives in group A using feature A

print("\n\tCancer\tNo Cancer")                                     # Header for group A confusion matrix
print("Actual C:", TPa, "\t  ", FNa)                               # Print TP and FN for group A
print("Actual N:", FPa, "\t  ", TNa)                               # Print FP and TN for group A

tpFpListB = []                                                     # Initialize list for (feature, TP-FP) inside group B
for f in group_B.drop(columns=["Sample"]).columns:                 # Loop through features within group B
    tp = group_B[group_B["Sample"] == "C"][f].sum()                # Count C with feature=1 in group B
    fp = group_B[group_B["Sample"] == "NC"][f].sum()               # Count NC with feature=1 in group B
    tpFpListB.append((f, tp - fp))                                 # Append (feature, TP-FP) for group B
tpFpListB.sort(key=lambda x: x[1], reverse=True)                   # Sort features by TP-FP for group B

print("\n\t\t\tTop 10 Features (Group B):")                        # Header for group B top features
for item in tpFpListB[:10]:                                        # Loop through top 10 features in group B
    print(item)                                                    # Print feature name and TP-FP value
B = tpFpListB[0][0]                                                # Select best feature for group B (B)

TPb = ((group_B["Sample"] == "C") & (group_B[B] == 1)).sum()       # True positives in group B using feature B
FPb = ((group_B["Sample"] == "NC") & (group_B[B] == 1)).sum()      # False positives in group B using feature B
TNb = ((group_B["Sample"] == "NC") & (group_B[B] == 0)).sum()      # True negatives in group B using feature B
FNb = ((group_B["Sample"] == "C") & (group_B[B] == 0)).sum()       # False negatives in group B using feature B

print("\n\tCancer\tNo Cancer")                                       # Header for group B confusion matrix
print("Actual C:", TPb, "\t  ", FNb)                               # Print TP and FN for group B
print("Actual N:", FPb, "\t  ", TNb)                               # Print FP and TN for group B

print("\n\t\t\tTwo-Level Decision Tree\n")                        # Header for decision tree
print(f"{F}")                                                     # Print root feature name
print("├─1─", A)                                                  # Branch when F=1: show A
print("│   ├─1→ C")                                               # If A=1 then Cancer
print("│   └─0→ NC")                                              # If A=0 then No Cancer
print("└─0─", B)                                                  # Branch when F=0: show B
print("    ├─1→ C")                                               # If B=1 then Cancer
print("    └─0→ NC")                                              # If B=0 then No Cancer

print("\nRules:")                                                 # Header for rules
print(f"If {F}=1: \nHas {A}=1 -> C \nelse -> NC\n")                      # Rule for F=1 path
print(f"If {F}=0: \nHas {B}=1 -> C \nelse -> NC\n")                      # Rule for F=0 path


test_samples = ["C1","C10","C30","NC5","NC15"]                    # List of samples to classify
original = pd.read_csv("mutations.csv")                           # Reload original CSV to keep sample names intact
original = original.rename(columns={"Unnamed: 0": "Sample"})      # Rename first column to "Sample" again
correct = 0                                                       # Counter for correct classifications
for s in test_samples:                                            # Loop through each test sample
    row = original[original["Sample"] == s].iloc[0]               # Get the row of that sample
    pred = None                                                   # Initialize predicted label
    if row[F]==1:                                                 # If sample has root feature F=1
        pred = "C" if row[A]==1 else "NC"                         # Then check feature A for classification
    else:                                                         # Else sample has F=0
        pred = "C" if row[B]==1 else "NC"                         # Then check feature B for classification
    truth = "C" if s.startswith("C") else "NC"                    # True label is based on sample name prefix
    print(f"{s}: predicted {pred}, truth {truth}")                # Print prediction and actual
    if pred == truth: correct += 1                                # Increment correct counter if prediction matches truth

print(f"\nCorrect classifications: {correct}/{len(test_samples)}")  # Print final accuracy on the test samples
