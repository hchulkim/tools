import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv("../data/lalonde_nsw.csv")

# check the data sample
print(data.head())

print(data.describe())


# 1 (a). Compute the average treatment effect of the policy on re78 for the whole sample.

ate = data["re78"][data["treat"] == 1].mean() - data["re78"][data["treat"] == 0].mean()

# 1 (b). Compute the average treatment effect of the treated

att = data["re78"][data["treat"] == 1].mean() - data["re78"][data["treat"] == 0].mean()

# Since the treatment group is randomly selected, ATT = ATE.

# 1 (c). est the null of τi = 0 for all i using a randomization test. N.B. Hold fixed the number of treated and control (e.g. assume the treatment count would be held fixed) and permute the labels randomly 1000 times – you do not need to fully do every permutation (there would be too many). Report the quantile that your estimate from the previous question falls.


# Randomization test
n_perm = 1000
ate_perm = np.zeros(n_perm)
for i in range(n_perm):
    perm = np.random.permutation(data["re78"])
    data_perm = data.copy()
    data_perm["re78"] = perm
    ate_perm[i] = (
        data_perm["re78"][data_perm["treat"] == 1].mean()
        - data_perm["re78"][data_perm["treat"] == 0].mean()
    )

quantile = np.mean(ate_perm < ate)  # p-value is about 0.003

# 1 (d). I did this regression using a separate R script.
