import math
import random
import textwrap
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate


random.seed(9)


def load_data(path: str) -> pd.DataFrame:
    """Step 1: read the CSV, normalize headers, and add the Class label."""
    df = pd.read_csv(path)
    sample_col = df.columns[0]
    df = df.rename(columns={sample_col: "Sample"})
    df["Class"] = df["Sample"].apply(label_from_sample)
    if df["Class"].isnull().any():
        missing = df[df["Class"].isnull()]["Sample"].tolist()
        raise ValueError(f"Unlabeled samples encountered: {missing[:5]}")
    return df


def label_from_sample(name: str) -> str:
    """Translate sample IDs like 'C1' or 'NC2' into class labels."""
    tag = str(name).strip().upper()
    if tag.startswith("C"):
        return "C"
    if tag.startswith("NC"):
        return "NC"
    return ""

def entropy(n_c: int, n_nc: int) -> float:
    """Compute Shannon entropy for a node containing counts of C and NC."""
    total = n_c + n_nc                     # Total samples 
    if total == 0:                         
        return 0.0
    value = 0.0
    for count in (n_c, n_nc):              
        if count == 0:                     
            continue
        p = count / total                  # Probability of this class
        value -= p * math.log2(p)          # Shannon entropy contribution
    return value                           


def node_entropy(df: pd.DataFrame) -> float:
    # Count how many 'C' and 'NC' samples exist and pass them into entropy()
    return entropy((df["Class"] == "C").sum(), (df["Class"] == "NC").sum())


def information_gain(df: pd.DataFrame, feature: str) -> float:
    """Calculate how much information is gained by splitting on one feature."""
    parent_entropy = node_entropy(df)            # Entropy before split (whole node)

    # Divide dataset into two parts depending on feature value

    left = df[df[feature] == 1]                  
    right = df[df[feature] == 0]                 # Subset where gene not mutated

    n = len(df)                                  # Total number of samples

    if n == 0:                                   # Empty means no info gain
        return 0.0

    # Compute entropy for each child node
    h_left = node_entropy(left)
    h_right = node_entropy(right)

    # Compute weighted entropy 
    weighted = (len(left) / n) * h_left + (len(right) / n) * h_right

    # Information gain = parent entropy − average child entropy
    return parent_entropy - weighted



# GENERAL HELPERS
 

def safe_ratio(num: float, denom: float) -> float:
    """Compute ratio safely to avoid division by zero."""
    return 0.0 if denom == 0 else num / denom


def majority_class(df: pd.DataFrame) -> str:
    """Return whichever class ('C' or 'NC') is most frequent in df."""
    counts = df["Class"].value_counts()      # Count class frequencies
    return counts.idxmax() if not counts.empty else "NC"  # Pick most common



# TREE STRUCTURE
 
# individual trees predict a sample
@dataclass
class TreeNode:
    # Represents a single decision node.
    feature: str | None                      # The feature used to split (None for leaf)
    prediction: str                          # The majority label in this node
    left: "TreeNode | None" = None           # Left child node (feature == 1)
    right: "TreeNode | None" = None          # Right child node (feature == 0)

    def predict(self, sample: pd.Series) -> str:
        node = self                          # Start from root
        while node.feature is not None:      # While not at a leaf node
            branch = sample[node.feature]    # Look up this sample’s features value
            
            node = node.left if branch == 1 else node.right
            if node is None:                 #
                break
        # Return class label stored at the leaf node
        return node.prediction if node else "NC"


 
# TREE BUILDER

class DepthTwoTreeBuilder:
    def __init__(self, features: list[str], subset_size: int) -> None:
        # list of all genes/features available
        # number of features to sample at each split 
        self.features = features
        self.subset_size = subset_size
        self.root_feature: str | None = None            # Record root feature
        self.child_splits: dict[str, str | None] = {"left": None, "right": None}  # Child splits for reporting

    def build(self, data: pd.DataFrame) -> TreeNode:
        """Start recursive building from the root."""
        return self._build_node(data, depth=0, branch=None)

    def _build_node(self, data: pd.DataFrame, depth: int, branch: str | None) -> TreeNode:
        # Determine the majority class label at this node
        prediction = majority_class(data)

        # Stop if:
        # - max depth reach is 2
        # - the node is empty
        # - the is node already pure
        if depth >= 2 or data.empty or data["Class"].nunique() == 1:
            if depth == 1 and branch:  # record missing split name
                self.child_splits.setdefault(branch, None)
            return TreeNode(feature=None, prediction=prediction)  # create leaf

        #FEATURE SELECTION
        subset = random.sample(self.features, self.subset_size)   # Randomly choose √n features
        best_feature, best_gain = self._best_feature(data, subset) # Evaluate each

        # If no improvement or invalid feature, stop and return leaf
        if best_feature is None or best_gain < 0:
            if depth == 1 and branch:
                self.child_splits[branch] = None
            return TreeNode(feature=None, prediction=prediction)

        # Save root and child feature names for reporting later
        if depth == 0:
            self.root_feature = best_feature
        elif depth == 1 and branch:
            self.child_splits[branch] = best_feature

        # Create a new node that splits on the chosen feature
        node = TreeNode(feature=best_feature, prediction=prediction)

        # Split data into two branches (feature present vs absent)
        left_data = data[data[best_feature] == 1]
        right_data = data[data[best_feature] == 0]

        # If we're currently at the root, record branch direction labels
        next_branch_left = "left" if depth == 0 else None
        next_branch_right = "right" if depth == 0 else None

        # Recursively build the left and right subtrees
        node.left = self._build_node(left_data, depth + 1, next_branch_left)
        node.right = self._build_node(right_data, depth + 1, next_branch_right)
        return node

    def _best_feature(self, data: pd.DataFrame, subset: list[str]) -> tuple[str | None, float]:
        """Return the feature with highest information gain."""
        best_feature = None
        best_gain = -1.0
        for feature in subset:                 # Evaluate each sampled feature
            gain = information_gain(data, feature)
            if gain > best_gain:               # Keep track of the best one
                best_feature = feature
                best_gain = gain
        return best_feature, best_gain



# RANDOM FOREST DATA

@dataclass
class RandomTree:
    # Holds a trained tree + its associated metadata
    root: TreeNode                            # Root node of this tree
    oob_samples: list[str]                   
    root_feature: str | None                  # Which feature was used at root
    child_features: dict[str, str | None]     # Left/right child splits


# random forest makes a prodiction for a single sample by getting votes from all trees
class RandomForest:
    def __init__(self, trees: list[RandomTree]) -> None:
        self.trees = trees                     # Store all trees in the forest

    def predict(self, sample: pd.Series) -> tuple[str, int, int]:
        """Aggregate votes from every tree for a given sample."""
        votes_c = votes_nc = 0  # vote counters

        # Loop through every tree and get its individual prediction
        for tree in self.trees:
            pred = tree.root.predict(sample)
            if pred == "C":   # if tree votes 'C', increment C votes
                votes_c += 1
            else: 
                votes_nc += 1

        # Final class = majority vote across all trees
        label = "C" if votes_c >= votes_nc else "NC"
        return label, votes_c, votes_nc


# WHERE BOOTSTRAP SAMPLING STARTS
# pick number of samples with replacement, and track which samples are left out
def bootstrap_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Generate bootstrap sample (with replacement) + track OOB samples."""
    draw = random.choices(df.index.tolist(), k=len(df))  # Sample n rows with replacement
    bootstrap_df = df.loc[draw].reset_index(drop=True)   # Create bootstrap dataset
    # Out-of-bag samples
    oob = sorted(set(df["Sample"]) - set(bootstrap_df["Sample"]))
    return bootstrap_df, oob


def build_depth_two_tree(df: pd.DataFrame, features: list[str]) -> RandomTree:
    """Train one small (depth-2) random decision tree."""
    subset_size = max(1, int(math.sqrt(len(features))))  # root n features per split
    builder = DepthTwoTreeBuilder(features, subset_size) # Initialize builder
    root = builder.build(df)                             # Build tree structure
    return RandomTree(
        root=root,
        oob_samples=[],                                  # OOB set will be added later
        root_feature=builder.root_feature,               
        child_features=builder.child_splits.copy(),      
    )

# build the random forest by creating multiple trees with bootstrapped data
def build_random_forest(full_df: pd.DataFrame, n_trees: int, features: list[str]) -> tuple[list[RandomTree], list[str]]:
    """Train multiple random trees to form the full forest."""
    trees: list[RandomTree] = []
    initial_oob: list[str] = []   # Keep OOB samples from first tree for evaluation
    for idx in range(n_trees):
        # Create bootstrap dataset and get its OOB samples
        sample_df, oob = bootstrap_sample(full_df)
        # Build a tree using this bootstrap
        tree = build_depth_two_tree(sample_df, features)
        # Assign OOB samples to the tree
        tree.oob_samples = oob
        if idx == 0:
            initial_oob = oob
        trees.append(tree)
    return trees, initial_oob


 
# REPORTING
 

def _fmt_feature(feature: str | None) -> str:
    """Return feature name or 'None' if no split feature."""
    return feature if feature is not None else "None"


def report_single_tree(tree: RandomTree) -> None:
    """Print readable details about one tree (root + child splits)."""
    print("PART 1 ─ Depth-2 Tree Details")
    print(f"out-of-bag size: {len(tree.oob_samples)}")  # Show how many OOB samples
    if tree.oob_samples:
        # Wrap long OOB list nicely across lines
        wrapped = textwrap.fill(", ".join(tree.oob_samples), width=100, subsequent_indent="  ")
        print(f"out-of-bag samples:\n  {wrapped}")
    else:
        print("out-of-bag samples: ∅")

    # Print which feature was chosen at root and its children
    print(f"root split: {_fmt_feature(tree.root_feature)}")
    child_rows = []
    for branch in ("left", "right"):
        child_rows.append([branch, _fmt_feature(tree.child_features.get(branch))])
    print("child splits")
    print(tabulate(child_rows, headers=["branch", "gene"], tablefmt="github"))


def summarize_forest(trees: list[RandomTree]) -> None:
    """Summarize all trees: how often each feature was used + OOB sizes."""
    root_counter = Counter()
    child_counter = Counter()
    oob_sizes = []

    # Collect usage counts and OOB sizes from each tree
    for tree in trees:
        if tree.root_feature:
            root_counter[tree.root_feature] += 1
        for feat in tree.child_features.values():
            if feat:
                child_counter[feat] += 1
        oob_sizes.append(len(tree.oob_samples))

    print("\nPART 1 ─ Random Forest Summary")

    # Root split usage summary
    if root_counter:
        print("\nroot split usage")
        print(tabulate(sorted(root_counter.items()), headers=["gene", "count"], tablefmt="github"))

    # Child split usage summary
    if child_counter:
        print("\nchild split usage")
        print(tabulate(sorted(child_counter.items()), headers=["gene", "count"], tablefmt="github"))

    # OOB sizes per tree
    oob_rows = [[idx, size] for idx, size in enumerate(oob_sizes, start=1)]
    print("\nout-of-bag sizes per tree")
    print(tabulate(oob_rows, headers=["tree #", "oob size"], tablefmt="github"))

    # Print average OOB size across all trees
    avg_oob = sum(oob_sizes) / len(oob_sizes)
    print(f"\navg oob size: {avg_oob:.2f}")


# classify samples and show votes
# for each sample, show the sample name, actual class, predicted class, and votes for C/NC
def classify_samples(forest: RandomForest, sample_index: pd.DataFrame, samples: list[str]) -> None:
    """Classify selected samples and show how many trees voted C/NC."""
    rows = []
    for name in samples:
        row = sample_index.loc[name]          # Find this sample in dataframe
        pred, votes_c, votes_nc = forest.predict(row)  # Predict via forest
        rows.append([name, row["Class"], pred, votes_c, votes_nc])
    print("\nPART 2 ─ Required Sample Classifications")
    print(tabulate(rows, headers=["sample", "actual", "rf prediction", "#C votes", "#NC votes"], tablefmt="github"))

# out of bag evaluation starts
# use the random forest to classify the OOB samples from the first tree
def evaluate_on_oob(forest: RandomForest, sample_index: pd.DataFrame, oob_samples: list[str]) -> None:
    """Evaluate model on out-of-bag samples and print confusion matrix."""
    tp = fp = tn = fn = 0

    # Predict each OOB sample using the forest
    for name in oob_samples:
        row = sample_index.loc[name]
        pred, _, _ = forest.predict(row)
        actual = row["Class"]

        # Compare predicted vs actual and update confusion counts
        if actual == "C" and pred == "C":
            tp += 1
        elif actual == "C" and pred == "NC":
            fn += 1
        elif actual == "NC" and pred == "C":
            fp += 1
        else:
            tn += 1

    # Print confusion table
    counts = [["TP", tp], ["FP", fp], ["TN", tn], ["FN", fn]]
    print("\nPART 2 ─ OOB Confusion (tree #1 set)")
    print(tabulate(counts, headers=["metric", "value"], tablefmt="github"))

    # Calculate key performance metrics
    metrics = [
        ["accuracy", safe_ratio(tp + tn, tp + tn + fp + fn)],
        ["sensitivity", safe_ratio(tp, tp + fn)],
        ["specificity", safe_ratio(tn, tn + fp)],
        ["precision", safe_ratio(tp, tp + fp)],
        ["miss rate", safe_ratio(fn, tp + fn)],
        ["false discovery rate", safe_ratio(fp, tp + fp)],
        ["false omission rate", safe_ratio(fn, fn + tn)],
    ]
    print(tabulate(metrics, headers=["metric", "value"], tablefmt="github"))


 
# MAIN
 

def main() -> None:
    """Run the complete workflow end-to-end."""
    # Step 1: Load dataset and prepare list of features
    df = load_data("mutations9.csv")  # CSV must have mutation features per sample
    features = [col for col in df.columns if col not in {"Sample", "Class"}]

    # Step 1 and 2: Build a forest of 15 small trees using bootstrapped data
    n_trees = 15
    trees, initial_oob = build_random_forest(df, n_trees, features)
    forest = RandomForest(trees)

    # Step 3 report: describe the first tree, then the whole forest
    report_single_tree(trees[0])
    summarize_forest(trees)

    # Step 4: Classify specific example samples and show votes
    sample_index = df.set_index("Sample")
    classify_samples(forest, sample_index, ["C1", "C10", "C15", "NC5", "NC15"])

    # Step 5: Evaluate forest performance using the first tree’s OOB samples
    evaluate_on_oob(forest, sample_index, initial_oob)


if __name__ == "__main__":
    main()
