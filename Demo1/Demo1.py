import pandas as pd  # used to read and handle data tables


#1. Load the data
path = "mutations.csv"
df = pd.read_csv(path)        


# rename first columb to sample for clarity
if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "Sample"})

df = df.reset_index(drop=True)   # reset row numbers to start from 0


# create a new column called "label" 
# if sample name starts with C then set its label to "C" and do the same with NC
# this will be used for the decision tree.
df["Label"] = ["C" if str(s).startswith("C") else "NC" for s in df["Sample"]]


# 2. Compute accuracy for each feature

# loop over every column name in the dataset (df.columns)
# keep only ones where all columns are 1 or 0.
features = [column for column in df.columns if column not in ("Sample", "Label")]
results = []                                                        # empty list to store each feature

for f in features:                                     # loop over every mutation column
    TP = ((df["Label"] == "C") & (df[f] == 1)).sum()   # cancer samples predicted as cancer TP
    FP = ((df["Label"] == "NC") & (df[f] == 1)).sum()  # non-cancer predicted as cancer FP
    TN = ((df["Label"] == "NC") & (df[f] == 0)).sum()  # non-cancer predicted correctly TN
    FN = ((df["Label"] == "C") & (df[f] == 0)).sum()   # cancer missed by prediction FN
    
    total = TP + FP + TN + FN                          # total number of samples
                                     
    acc = (TP + TN) / total                            # accuracy formula off the wiki 

    results.append((f, acc))                           # store feature name and accuracy


# sort all features from highest to lowest accuracy x[1] is the acutal number value in the tuple 
# x[0] is just the feature name
results.sort(key=lambda x: x[1], reverse=True)


# print the top 10 most accurate features
print("\nTop 10 Features by Accuracy:\n")
print(f"{'Feature':<60} Accuracy") # <60 means left align in a space of 60 characters

for f, a in results[:10]:
    print(f"{f:<25} {a:.6f}")


# get feature with the highest accuracy
best_feature = results[0][0]
print("\nMost accurate feature:", best_feature)


# 3. Build confusion matrix

# looks into the df column for best feature and counts the TP, FP, TN, FN

TP = ((df["Label"] == "C") & (df[best_feature] == 1)).sum()   # True Positive
FP = ((df["Label"] == "NC") & (df[best_feature] == 1)).sum()  # False Positive
TN = ((df["Label"] == "NC") & (df[best_feature] == 0)).sum()  # True Negative
FN = ((df["Label"] == "C") & (df[best_feature] == 0)).sum()   # False Negative


# print confusion matrix
print("\nConfusion Matrix:\n")
print("\t\tPred C\tPred NC")
print(f"Actual C:\t{TP:>6}\t{FN:>6}") # >6 means right align in a space of 6 characters
print(f"Actual NC:\t{FP:>6}\t{TN:>6}")


# 4. Compute performance metrics

# 1. Total number of samples used do else 0 to avoid division by zero
total = TP + FP + TN + FN

# 2. Accuracy formula: https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
accuracy = (TP + TN) / total

# 3. Sensitivity formula: https://en.wikipedia.org/wiki/Sensitivity_and_specificity#In_binary_classification
sensitivity = TP / (TP + FN)

# 4. Specificity formula: https://en.wikipedia.org/wiki/Sensitivity_and_specificity#In_binary_classification
specificity = TN / (TN + FP)

# 5. Precision formula: https://en.wikipedia.org/wiki/Precision_and_recall#In_binary_classification
precision = TP / (TP + FP)

# 6. Miss Rate formula: https://en.wikipedia.org/wiki/Sensitivity_and_specificity#In_binary_classification
miss_rate = FN / (TP + FN)

# 7. False Discovery Rate formula: https://en.wikipedia.org/wiki/Precision_and_recall#In_binary_classification
false_discovery = FP / (TP + FP)

# 8. False Omission Rate formula: https://en.wikipedia.org/wiki/Precision_and_recall#In_binary_classification
false_omission = FN / (FN + TN)


# print all metrics
print("\nClassification Metrics:\n")
print(f"Accuracy:             {accuracy:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity:          {specificity:.6f}")
print(f"Precision:            {precision:.6f}")
print(f"Miss Rate:            {miss_rate:.6f}")
print(f"False Discovery Rate: {false_discovery:.6f}")
print(f"False Omission Rate:  {false_omission:.6f}")
