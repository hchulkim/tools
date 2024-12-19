import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pyblp

pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

## Questions

### 1. Describe the data
product = pd.read_csv("../Data/products.csv")

print(product.sample(10))


### 2. Compute market shares
product["market_size"] = product["city_population"] * 90
product["market_share"] = product["servings_sold"] / product["market_size"]

product["outside_share"] = 1 - product.groupby("market")["market_share"].transform(
    "sum"
)
product[["market_share", "outside_share"]].describe()


### 3. Estimate the pure logit model with OLS
product["logit_delta"] = np.log(product["market_share"] / product["outside_share"])

statsmodels_ols = smf.ols("logit_delta ~ 1 + mushy + price_per_serving", product)
statsmodels_results = statsmodels_ols.fit(cov_type="HC0")
statsmodels_results.summary2().tables[1]


### 4. Run the same regression with PyBLP
product = product.rename(
    columns={
        "market": "market_ids",
        "product": "product_ids",
        "market_share": "shares",
        "price_per_serving": "prices",
    }
)
product["demand_instruments0"] = product["prices"]
ols_problem = pyblp.Problem(pyblp.Formulation("1 + mushy + prices"), product)
ols_problem

ols_results = ols_problem.solve(method="1s")
ols_results


### 5. Add market and product fixed effects
ols_fe_problem = pyblp.Problem(
    pyblp.Formulation("0 + prices", absorb="C(market_ids) + C(product_ids)"), product
)
ols_fe_problem

ols_fe_results = ols_fe_problem.solve(method="1s")
ols_fe_results


### 6. Add an instrument for price
first_stage = smf.ols(
    "prices ~ 0 + price_instrument + C(market_ids) + C(product_ids)", product
)
first_stage_results = first_stage.fit(cov_type="HC0")
first_stage_results.summary2().tables[1].sort_index(ascending=False)

product = product.drop(columns="demand_instruments0").rename(
    columns={"price_instrument": "demand_instruments0"}
)
iv_problem = pyblp.Problem(
    pyblp.Formulation("0 + prices", absorb="C(market_ids) + C(product_ids)"), product
)
iv_problem

iv_results = iv_problem.solve(method="1s")
iv_results


### 7. Cut a price in half and see what happens
counterfactual_market = "C01Q2"
counterfactual_data = product.loc[
    product["market_ids"] == counterfactual_market,
    ["product_ids", "mushy", "prices", "shares"],
]
counterfactual_data

counterfactual_data["new_prices"] = counterfactual_data["prices"]
counterfactual_data.loc[
    counterfactual_data["product_ids"] == "F1B04", "new_prices"
] /= 2
counterfactual_data["new_shares"] = iv_results.compute_shares(
    market_id=counterfactual_market, prices=counterfactual_data["new_prices"]
)
counterfactual_data["iv_change"] = (
    100
    * (counterfactual_data["new_shares"] - counterfactual_data["shares"])
    / counterfactual_data["shares"]
)
counterfactual_data


### 8. Compute demand elasticities
iv_elasticities = iv_results.compute_elasticities(market_id=counterfactual_market)
pd.DataFrame(iv_elasticities)


## Exericse 2

# read in the demographics csv data.
demographics = pd.read_csv("../Data/demographics.csv")

print(demographics.sample(10))

demographics["log_income"] = np.log(demographics["quarterly_income"])

### 1. Describe cross-market variation

choice_variation = product.groupby("market_ids", as_index=False).agg(
    **{
        "products": ("product_ids", "count"),
        "mushy_mean": ("mushy", "mean"),
        "mushy_std": ("mushy", "std"),
        "prices_mean": ("prices", "mean"),
        "prices_std": ("prices", "std"),
    }
)
choice_variation.describe()

demographics = demographics.rename(columns={"market": "market_ids"})
demographics.sample(n=5, random_state=0)

demographic_variation = demographics.groupby("market_ids", as_index=False).agg(
    **{
        "log_income_mean": ("log_income", "mean"),
        "log_income_std": ("log_income", "std"),
    }
)
demographic_variation.describe()

## 2. Estimate a parameter on mushy x log income
agent_data = (
    demographics[["market_ids", "log_income"]]
    .groupby("market_ids", as_index=False)
    .sample(n=1000, replace=True, random_state=0)
)
agent_data[["nodes0", "nodes1", "nodes2"]] = np.random.default_rng(seed=0).normal(
    size=(len(agent_data), 3)
)
agent_data["weights"] = 1 / agent_data.groupby("market_ids").transform("size")
agent_data.sample(n=5, random_state=0)

product_data = product.merge(
    demographic_variation[["market_ids", "log_income_mean"]], on="market_ids"
)
product_data["demand_instruments1"] = (
    product_data["log_income_mean"] * product_data["mushy"]
)
