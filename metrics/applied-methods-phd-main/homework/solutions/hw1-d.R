

# 1 (d)

# load necessary packages

library(renv)

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, fixest, texreg)


# read in the data
data <- read_csv("data/lalonde_nsw.csv")


# Run a regression using robust standard errors

reg <- feols(re78 ~ treat, data = data)

# The reg result
screenreg(reg)

# Use summary to get p-value
summary(reg)

# It is about 0.004 - ish. So it is very similar.
