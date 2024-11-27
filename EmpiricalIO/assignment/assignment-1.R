
##
# Assigment 1 solutions
##

if (!require(pacman)) install.packages("pacman")
p_load(here, tidyverse, evd)

set.seed(1)

data <- tibble(i = rep(1:1000, each = 2),
             k = rep(1:2, 1000),
             x = rep(0:1, 1000),
             e = rgumbel(2000))

beta <- 0.2

# adding latent variable using equation: \beta*x_k + \varepsilon_{ik}
data <- data %>% mutate(latent = beta * x + e)

# compute y by comparing the latent values k=1,2 for each i
data <- data %>% 
  group_by(i) %>% 
  mutate(y = dense_rank(desc(latent))) %>% 
  mutate(y = if_else(y==1, 1, 0)) %>% 
  ungroup()

compute_loglikelihood_ad <- function(par, data) {
  # Add a column with the probability for each k = 1
  data_with_probs <- data %>% 
    group_by(i) %>% 
    mutate(
      p1 = exp(par * x[k ==1]) / (exp(par * x[k == 1]) + exp(par * x[k == 2]))
    ) %>% 
    ungroup()
  # Compute the log likelihood using the formula
  log_likelihood <- data_with_probs %>% 
    filter(k == 1) %>% 
    summarise(
     loglik = (1/1000)*sum(y * log(p1) + (1-y) * log(1-p1)) 
    ) %>% 
    pull(loglik)
  
  return(log_likelihood)
}

# test the function
compute_loglikelihood_ad(1, data)

# find and report the beta that maximizes log likelihood for the simulated data.
optim(par = 0,
      fn = compute_loglikelihood_ad,
      method = "Brent",
      lower = -1,
      upper = 1,
      control = list(fnscale = -1),
      data = data)
