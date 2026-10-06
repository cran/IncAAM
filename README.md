# IncAAM

**Version 1.08**

**IncAAM** is an R package for Bayesian age-at-maturity analysis based on annual increment-width growth series, such as otolith increment measurements.

The package fits a hierarchical nonlinear mixed-effects model with a negative exponential growth curve and fish-level varying parameters. Age-at-maturity estimates are derived using two complementary approaches:

- **TR (tangent-ratio threshold)**, based on the posterior distribution of the exponential decay parameter `c`;
- **Jerk**, calculated numerically from posterior fitted curves.

IncAAM also calculates posterior threshold-crossing probabilities by increment number and fits a logistic crossing ogive, providing an estimated **ogive A50**.

## Installation

From a local source package:

```r
install.packages("IncAAM_0.1.0.tar.gz", repos = NULL, type = "source")
```

Or, from the package source directory:

```r
devtools::install()
```

## Dependencies

The model uses `brms` with the `cmdstanr` backend. Install the R dependencies first, then install CmdStan if it is not already available:

```r
install.packages(c(
  "brms", "cmdstanr", "dplyr", "tidyr", "purrr", "tibble",
  "ggplot2", "posterior", "minpack.lm", "bayesplot"
))
cmdstanr::install_cmdstan()
```

## Example

```r
library(IncAAM)

dfx <- read.csv("Combined_data_clean.csv") |>
  dplyr::filter(
    Species == "plaice",
    Sex == "female",
    Axis == "section"
  )

fit <- IncAAM(
  data = dfx,
  FishID = "FishID",
  Inc_num = "Inc_num",
  Inc_width = "Inc_width_mm",
  sex = "female",
  known_A50 = 7.6,
  n_iter = 1500,
  n_warmup = 750,
  n_keepdraws = 500,
  n_adapt_delta = 0.995,
  n_logc_sd = 0.1
)

fit$summary
fit$crossing_ogive
fit$ogive_summary
```

The function expects one analysis group at a time; filter variables such as species, sex, stock, or anatomical axis before calling `IncAAM()`.


## Methodological reference

Details of the model, method, outputs and applications are provided in Campana, Smoliński, Morrongiello and Black (2026, in press). Users should consult this publication for the methodological background and interpretation of IncAAM outputs.

### Output definitions

- **Crossing ogive:** the probability that the Tangent Ratio (TR) has fallen below the defined TR threshold at a given increment number, and thus indicates sexual maturity according to the selected threshold.
- **Ogive A50:** the estimated increment number at which the fitted crossing-ogive probability reaches 50%, i.e. the estimated A50.
- **TR index:** the estimated age at 50% maturity when the selected TR threshold is the maturity criterion calibrated for that purpose.
- **Jerk index:** the increment number corresponding to the maximum jerk. Its relationship to a specific proportion or stage of sexual maturity is currently not defined.
