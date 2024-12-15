import numpy as np
import pandas as pd
import pyblp
import statsmodels.formula.api as smf

pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

# download the data

product = pd.read_csv("data/products.csv")

print(product.describe)
