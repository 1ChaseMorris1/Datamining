import pandas as pd  # simple table utilities
from sklearn.model_selection import KFold  # reproducible 3-fold splits
from sklearn.tree import DecisionTreeClassifier  # quick decision tree implementation

# 1. Load the data -----------------------------------------------------------
path = "mutations.csv"
df = pd.read_csv(path)

# rename the sample column if it was stored without a header
if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "Sample"})

df = df.reset_index(drop=True)

# create label column based on sample naming convention (C = cancer, NC = non-cancer)
df["Label"] = ["C" if str(sample).startswith("C") else "NC" for sample in df["Sample"]]

# grab only the mutation columns (binary features)
features = [column for column in df.columns if column not in ("Sample", "Label")]

# STEP 3: 3-fold cross-validation with consistent shuffle -------------------
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
fold_metrics = []

for fold_number, (train_index, test_index) in enumerate(kfold.split(df), start=1):
    print("=" * 72)
    print(f"FOLD {fold_number}")
    print("=" * 72)

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples: {len(test_df)}\n")

    # STEP 1: fit a decision tree on the training subset
    tree = DecisionTreeClassifier(random_state=42, criterion="entropy")
    tree.fit(train_df[features], train_df["Label"])

    # report the most important splits (root, and first child on each branch)
    tree_structure = tree.tree_

    root_feature_index = tree_structure.feature[0]
    if root_feature_index != -2:
        root_feature_name = features[root_feature_index]
    else:
        root_feature_name = "No split (tree stayed a single leaf)"

    left_child_index = tree_structure.children_left[0]
    right_child_index = tree_structure.children_right[0]

    if left_child_index != -1:
        left_feature_index = tree_structure.feature[left_child_index]
        if left_feature_index != -2:
            left_feature_name = features[left_feature_index]
        else:
            left_feature_name = "Leaf (no further split)"
    else:
        left_feature_name = "Leaf (no further split)"

    if right_child_index != -1:
        right_feature_index = tree_structure.feature[right_child_index]
        if right_feature_index != -2:
            right_feature_name = features[right_feature_index]
        else:
            right_feature_name = "Leaf (no further split)"
    else:
        right_feature_name = "Leaf (no further split)"

    print("Best Root Feature (F):", root_feature_name)
    print("Best Feature for Group A (root branch value <= 0.5):", left_feature_name)
    print("Best Feature for Group B (root branch value > 0.5):", right_feature_name)
    print()

    # STEP 2: classify the held-out test subset
    y_test = test_df["Label"]
    y_pred = tree.predict(test_df[features])

    tp = int(((y_test == "C") & (y_pred == "C")).sum())
    fp = int(((y_test == "NC") & (y_pred == "C")).sum())
    tn = int(((y_test == "NC") & (y_pred == "NC")).sum())
    fn = int(((y_test == "C") & (y_pred == "NC")).sum())

    print("Confusion Matrix:")
    print("\t\tActual C\tActual NC")
    print(f"Predicted C\t{tp:>6}\t{fp:>6}")
    print(f"Predicted NC\t{fn:>6}\t{tn:>6}")
    print()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    miss_rate = fn / (tp + fn) if (tp + fn) else float("nan")
    false_discovery = fp / (tp + fp) if (tp + fp) else float("nan")
    false_omission = fn / (fn + tn) if (fn + tn) else float("nan")

    print("Decision Tree Performance Metrics For Test Set")
    print("=" * 72)
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Sensitivity:          {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Miss Rate:            {miss_rate:.4f}")
    print(f"False Discovery Rate: {false_discovery:.4f}")
    print(f"False Omission Rate:  {false_omission:.4f}")
    print()

    fold_metrics.append(
        {
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Miss Rate": miss_rate,
            "False Discovery Rate": false_discovery,
            "False Omission Rate": false_omission,
        }
    )

# STEP 3: report average metrics across folds --------------------------------
if fold_metrics:
    print("=" * 72)
    print("AVERAGE METRICS ACROSS 3 FOLDS")
    print("=" * 72)

    metric_names = [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "Miss Rate",
        "False Discovery Rate",
        "False Omission Rate",
    ]

    for metric_name in metric_names:
        average_value = sum(metrics[metric_name] for metrics in fold_metrics) / len(fold_metrics)
        print(f"{metric_name}: {average_value:.4f}")
