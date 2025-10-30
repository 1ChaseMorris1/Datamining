import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 
# Load the dataset and label each sample.
# Samples that start with "C" are cancer (C); the rest are non-cancer (NC).
# 
data = pd.read_csv("mutations.csv")
data = data.rename(columns={"Unnamed: 0": "Sample"})
data["Label"] = data["Sample"].apply(lambda x: "C" if str(x).startswith("C") else "NC")
features = [f for f in data.columns if f not in ("Sample", "Label")]

# 
# Calculate TP-FP score for one feature.
# This measures how much a feature helps separate cancer from non-cancer.
# 
def tp_fp_score(feature, df):
    """Measures how well a feature separates cancer from non-cancer using TP - FP."""
    tp = ((df["Label"] == "C") & (df[feature] == 1)).sum()
    fp = ((df["Label"] == "NC") & (df[feature] == 1)).sum()
    return tp - fp

# 
# Calculate F(s,t) = 2 * PL * PR * Q(s|t) for one feature.
# This version rewards splits that are both balanced and pure.
# 
def f_score(feature, df):
    """Calculates the F(s,t) score which rewards balanced and pure splits."""
    n_total = len(df)
    if n_total == 0:
        return 0

    left = df[df[feature] == 1]
    right = df[df[feature] == 0]
    n_left, n_right = len(left), len(right)
    if n_left == 0 or n_right == 0:
        return 0

    PL = n_left / n_total
    PR = n_right / n_total

    pC_L = (left["Label"] == "C").mean()
    pC_R = (right["Label"] == "C").mean()
    pNC_L = (left["Label"] == "NC").mean()
    pNC_R = (right["Label"] == "NC").mean()

    Q = abs(pC_L - pC_R) + abs(pNC_L - pNC_R)
    return 2 * PL * PR * Q

# 
# Build a small 2-level decision tree using the TP-FP method.
# It picks the best root and two child features based on TP-FP scores.
# 
def build_tree_tp_fp(df):
    """Finds the best root and child features using TP-FP scores."""
    best_root = None
    best_score = -9999
    for f in features:
        score = tp_fp_score(f, df)
        if score > best_score:
            best_score = score
            best_root = f

    left = df[df[best_root] == 1]
    right = df[df[best_root] == 0]

    best_left, best_right = None, None
    best_L, best_R = -9999, -9999
    for f in features:
        sL = tp_fp_score(f, left)
        sR = tp_fp_score(f, right)
        if sL > best_L:
            best_L = sL
            best_left = f
        if sR > best_R:
            best_R = sR
            best_right = f

    return best_root, best_left, best_right

# 
# Build a small 2-level decision tree using the F(s,t) method.
# This works the same way as TP-FP but uses the F function instead.
# 
def build_tree_f(df):
    """Finds the best root and child features using the F(s,t) formula."""
    best_root = None
    best_score = -1
    for f in features:
        score = f_score(f, df)
        if score > best_score:
            best_score = score
            best_root = f

    left = df[df[best_root] == 1]
    right = df[df[best_root] == 0]

    best_left, best_right = None, None
    best_L, best_R = -1, -1
    for f in features:
        sL = f_score(f, left)
        sR = f_score(f, right)
        if sL > best_L:
            best_L = sL
            best_left = f
        if sR > best_R:
            best_R = sR
            best_right = f

    return best_root, best_left, best_right

# 
# Evaluate a model using 3-fold cross-validation.
# It trains on two-thirds of the data and tests on one-third, repeating 3 times.
# Prints average accuracy and related statistics.
# 
def evaluate(df, builder, name):
    """Runs 3-fold cross-validation and prints average results."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    print(f"\n{name} Model Results:")
    fold = 1
    for train_i, test_i in kf.split(df):
        train = df.iloc[train_i]
        test = df.iloc[test_i]

        root, A, B = builder(train)

        preds, truth = [], []
        for _, row in test.iterrows():
            if row[root] == 1:
                pred = "C" if row[A] == 1 else "NC"
            else:
                pred = "C" if row[B] == 1 else "NC"
            preds.append(pred)
            truth.append(row["Label"])

        preds = np.array(preds)
        truth = np.array(truth)

        tp = np.sum((truth == "C") & (preds == "C"))
        fp = np.sum((truth == "NC") & (preds == "C"))
        tn = np.sum((truth == "NC") & (preds == "NC"))
        fn = np.sum((truth == "C") & (preds == "NC"))

        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0

        print(f"\nFold {fold}:")
        print(f" Root: {root}")
        print(f" Left (F=1): {A}")
        print(f" Right (F=0): {B}")
        print(f" Accuracy={acc:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}  Precision={prec:.4f}")

        results.append((acc, sens, spec, prec))
        fold += 1

    avg = np.mean(results, axis=0)
    print(f"\nAverage ({name}):")
    print(f" Accuracy:    {avg[0]:.4f}")
    print(f" Sensitivity: {avg[1]:.4f}")
    print(f" Specificity: {avg[2]:.4f}")
    print(f" Precision:   {avg[3]:.4f}")
    return avg

# 
# Run both the TP-FP and F(s,t) models to compare them.
# 
tp_fp_avg = evaluate(data, build_tree_tp_fp, "TP-FP")
f_func_avg = evaluate(data, build_tree_f, "F(s,t)")

# 
# Compare average results from both models side by side.
# 
print("\nModel Comparison:")
print(f"{'Metric':<15} | {'TP-FP Tree':>10} | {'F(s,t) Tree':>10}")
print("-" * 42)
metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
for i in range(4):
    print(f"{metrics[i]:<15} | {tp_fp_avg[i]:>10.4f} | {f_func_avg[i]:>10.4f}")

print("\nDifference (F(s,t) - TP-FP):")
for i in range(4):
    diff = f_func_avg[i] - tp_fp_avg[i]
    print(f"{metrics[i]:<15} | {diff:>10.4f}")

# 
# Final classification rules for how the F(s,t) model makes predictions.
# 
print("\nDecision Rules for F(s,t) Model:")
print("If F=1 → check A → if A=1 → Cancer, else → Non-Cancer")
print("If F=0 → check B → if B=1 → Cancer, else → Non-Cancer\n")
