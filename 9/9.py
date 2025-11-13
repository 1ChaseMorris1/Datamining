import math
import random
import textwrap
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate


random.seed(9)


def load_data(path: str) -> pd.DataFrame:
    """Step 1: read the CSV, rename the sample column, and add C/NC labels."""
    df = pd.read_csv(path)
    sample_col = df.columns[0]
    df = df.rename(columns={sample_col: "Sample"})
    df["Class"] = df["Sample"].apply(label_from_sample)
    if df["Class"].isnull().any():
        missing = df[df["Class"].isnull()]["Sample"].tolist()
        raise ValueError(f"Unlabeled samples encountered: {missing[:5]}")
    return df


def label_from_sample(name: str) -> str:
    """Map sample IDs like C1/NC2 to their class letter."""
    tag = str(name).strip().upper()
    if tag.startswith("C"):
        return "C"
    if tag.startswith("NC"):
        return "NC"
    return ""


def entropy(n_c: int, n_nc: int) -> float:
    """Shannon entropy helper for any node distribution."""
    total = n_c + n_nc
    if total == 0:
        return 0.0
    value = 0.0
    for count in (n_c, n_nc):
        if count == 0:
            continue
        p = count / total
        value -= p * math.log2(p)
    return value


def node_entropy(df: pd.DataFrame) -> float:
    """Convenience wrapper to compute entropy for a dataframe node."""
    return entropy((df["Class"] == "C").sum(), (df["Class"] == "NC").sum())


def information_gain(df: pd.DataFrame, feature: str) -> float:
    """Compute gain for splitting the provided node on the given feature."""
    parent_entropy = node_entropy(df)
    left = df[df[feature] == 1]
    right = df[df[feature] == 0]
    n = len(df)
    if n == 0:
        return 0.0
    h_left = node_entropy(left)
    h_right = node_entropy(right)
    weighted = (len(left) / n) * h_left + (len(right) / n) * h_right
    return parent_entropy - weighted


def safe_ratio(num: float, denom: float) -> float:
    """Avoid divide-by-zero when deriving evaluation metrics."""
    return 0.0 if denom == 0 else num / denom


def majority_class(df: pd.DataFrame) -> str:
    """Return the most common class label in a node."""
    counts = df["Class"].value_counts()
    return counts.idxmax() if not counts.empty else "NC"


@dataclass
class TreeNode:
    feature: str | None
    prediction: str
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, sample: pd.Series) -> str:
        """Traverse the depth-two tree to produce a label for one sample."""
        node = self
        while node.feature is not None:
            branch = sample[node.feature]
            node = node.left if branch == 1 else node.right
            if node is None:
                break
        return node.prediction if node else "NC"


class GeneAlias:
    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}
        self._counter = 1

    def alias(self, feature: str | None) -> str:
        """Assign short gene ids so tabular output stays compact."""
        if feature is None:
            return "None"
        if feature not in self._mapping:
            self._mapping[feature] = f"gene{self._counter}"
            self._counter += 1
        return self._mapping[feature]

    def legend(self) -> list[tuple[str, str]]:
        """Return the alias legend for final reporting."""
        return sorted((alias, feat) for feat, alias in self._mapping.items())


class DepthTwoTreeBuilder:
    def __init__(self, features: list[str], subset_size: int) -> None:
        self.features = features
        self.subset_size = subset_size
        self.root_feature: str | None = None
        self.child_splits: dict[str, str | None] = {"left": None, "right": None}

    def build(self, data: pd.DataFrame) -> TreeNode:
        """Step 2: build a depth-two tree using bootstrap data."""
        return self._build_node(data, depth=0, branch=None)

    def _build_node(self, data: pd.DataFrame, depth: int, branch: str | None) -> TreeNode:
        prediction = majority_class(data)
        if depth >= 2 or data.empty or data["Class"].nunique() == 1:
            if depth == 1 and branch:
                self.child_splits.setdefault(branch, None)
            return TreeNode(feature=None, prediction=prediction)

        # Step 2a: randomly sample sqrt(n) features for this split.
        subset = random.sample(self.features, self.subset_size)
        best_feature, best_gain = self._best_feature(data, subset)
        if best_feature is None or best_gain < 0:
            if depth == 1 and branch:
                self.child_splits[branch] = None
            return TreeNode(feature=None, prediction=prediction)

        if depth == 0:
            self.root_feature = best_feature
        elif depth == 1 and branch:
            self.child_splits[branch] = best_feature

        node = TreeNode(feature=best_feature, prediction=prediction)
        # Step 2b: recurse one level deeper on each branch.
        left_data = data[data[best_feature] == 1]
        right_data = data[data[best_feature] == 0]
        next_branch_left = "left" if depth == 0 else None
        next_branch_right = "right" if depth == 0 else None
        node.left = self._build_node(left_data, depth + 1, next_branch_left)
        node.right = self._build_node(right_data, depth + 1, next_branch_right)
        return node

    def _best_feature(self, data: pd.DataFrame, subset: list[str]) -> tuple[str | None, float]:
        best_feature = None
        best_gain = -1.0
        for feature in subset:
            gain = information_gain(data, feature)
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
        return best_feature, best_gain


@dataclass
class RandomTree:
    root: TreeNode
    oob_samples: list[str]
    root_feature: str | None
    child_features: dict[str, str | None]


class RandomForest:
    def __init__(self, trees: list[RandomTree]) -> None:
        self.trees = trees

    def predict(self, sample: pd.Series) -> tuple[str, int, int]:
        """Aggregate votes from every tree for a single sample."""
        votes_c = votes_nc = 0
        for tree in self.trees:
            pred = tree.root.predict(sample)
            if pred == "C":
                votes_c += 1
            else:
                votes_nc += 1
        label = "C" if votes_c >= votes_nc else "NC"
        return label, votes_c, votes_nc


def bootstrap_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Step 1a: create bootstrap data and collect associated OOB samples."""
    draw = random.choices(df.index.tolist(), k=len(df))
    bootstrap_df = df.loc[draw].reset_index(drop=True)
    oob = sorted(set(df["Sample"]) - set(bootstrap_df["Sample"]))
    return bootstrap_df, oob


def build_depth_two_tree(df: pd.DataFrame, features: list[str]) -> RandomTree:
    """Apply the depth-two builder to one bootstrap dataset."""
    subset_size = max(1, int(math.sqrt(len(features))))
    builder = DepthTwoTreeBuilder(features, subset_size)
    root = builder.build(df)
    return RandomTree(
        root=root,
        oob_samples=[],
        root_feature=builder.root_feature,
        child_features=builder.child_splits.copy(),
    )


def build_random_forest(full_df: pd.DataFrame, n_trees: int, features: list[str]) -> tuple[list[RandomTree], list[str]]:
    """Step 2: assemble the requested number of random trees."""
    trees: list[RandomTree] = []
    initial_oob: list[str] = []
    for idx in range(n_trees):
        sample_df, oob = bootstrap_sample(full_df)
        tree = build_depth_two_tree(sample_df, features)
        tree.oob_samples = oob
        if idx == 0:
            initial_oob = oob
        trees.append(tree)
    return trees, initial_oob


def report_single_tree(tree: RandomTree, aliaser: GeneAlias) -> None:
    """Print the detailed stats the worksheet requests for one tree."""
    print("PART 1 ─ Depth-2 Tree Details")
    print(f"out-of-bag size: {len(tree.oob_samples)}")
    if tree.oob_samples:
        wrapped = textwrap.fill(
            ", ".join(tree.oob_samples),
            width=100,
            subsequent_indent="  "
        )
        print(f"out-of-bag samples:\n  {wrapped}")
    else:
        print("out-of-bag samples: ∅")
    print(f"root split: {aliaser.alias(tree.root_feature)}")
    child_rows = []
    for branch in ("left", "right"):
        child_rows.append([branch, aliaser.alias(tree.child_features.get(branch))])
    print("child splits")
    print(tabulate(child_rows, headers=["branch", "gene"], tablefmt="github"))


def summarize_forest(trees: list[RandomTree], aliaser: GeneAlias) -> None:
    """Summarize root/child usage plus OOB size stats across the forest."""
    root_counter = Counter()
    child_counter = Counter()
    oob_sizes = []
    for tree in trees:
        if tree.root_feature:
            root_counter[aliaser.alias(tree.root_feature)] += 1
        for feat in tree.child_features.values():
            if feat:
                child_counter[aliaser.alias(feat)] += 1
        oob_sizes.append(len(tree.oob_samples))

    print("\nPART 1 ─ Random Forest Summary")
    if root_counter:
        print("\nroot split usage")
        print(tabulate(sorted(root_counter.items()), headers=["gene", "count"], tablefmt="github"))
    if child_counter:
        print("\nchild split usage")
        print(tabulate(sorted(child_counter.items()), headers=["gene", "count"], tablefmt="github"))

    oob_rows = [[idx, size] for idx, size in enumerate(oob_sizes, start=1)]
    print("\nout-of-bag sizes per tree")
    print(tabulate(oob_rows, headers=["tree #", "oob size"], tablefmt="github"))
    avg_oob = sum(oob_sizes) / len(oob_sizes)
    print(f"\navg oob size: {avg_oob:.2f}")


def classify_samples(forest: RandomForest, sample_index: pd.DataFrame, samples: list[str]) -> None:
    """Step 3: run the required C/NC predictions with vote counts."""
    rows = []
    for name in samples:
        row = sample_index.loc[name]
        pred, votes_c, votes_nc = forest.predict(row)
        rows.append([name, row["Class"], pred, votes_c, votes_nc])
    print("\nPART 2 ─ Required Sample Classifications")
    print(tabulate(rows, headers=["sample", "actual", "rf prediction", "#C votes", "#NC votes"], tablefmt="github"))


def evaluate_on_oob(forest: RandomForest, sample_index: pd.DataFrame, oob_samples: list[str]) -> None:
    """Step 4: score the initial OOB cohort and derive metrics."""
    tp = fp = tn = fn = 0
    for name in oob_samples:
        row = sample_index.loc[name]
        pred, _, _ = forest.predict(row)
        actual = row["Class"]
        if actual == "C" and pred == "C":
            tp += 1
        elif actual == "C" and pred == "NC":
            fn += 1
        elif actual == "NC" and pred == "C":
            fp += 1
        else:
            tn += 1

    counts = [["TP", tp], ["FP", fp], ["TN", tn], ["FN", fn]]
    print("\nPART 2 ─ OOB Confusion (tree #1 set)")
    print(tabulate(counts, headers=["metric", "value"], tablefmt="github"))

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


def report_aliases(aliaser: GeneAlias) -> None:
    """List the full feature names so the short ids are interpretable."""
    legend = aliaser.legend()
    if not legend:
        return
    print("\ngene legend")
    for alias, feat in legend:
        print(f"{alias}: {feat}")


def main() -> None:
    """Orchestrate the full workflow for Activity 9 Parts 1 and 2."""
    # Step 0: load, clean, and index the dataset once.
    df = load_data("mutations9.csv")
    features = [col for col in df.columns if col not in {"Sample", "Class"}]
    aliaser = GeneAlias()

    # Step 1-2: build a forest of depth-two trees using bootstrap samples.
    n_trees = 15
    trees, initial_oob = build_random_forest(df, n_trees, features)
    forest = RandomForest(trees)

    # Step 1 reporting: detail the first tree plus forest-level summary.
    report_single_tree(trees[0], aliaser)
    summarize_forest(trees, aliaser)

    sample_index = df.set_index("Sample")
    # Step 3: classify the specified samples with majority voting.
    classify_samples(forest, sample_index, ["C1", "C10", "C15", "NC5", "NC15"])
    # Step 4: evaluate against the first-tree OOB samples and document aliases.
    evaluate_on_oob(forest, sample_index, initial_oob)
    report_aliases(aliaser)


if __name__ == "__main__":
    main()
