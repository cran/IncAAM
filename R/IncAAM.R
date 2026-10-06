#' IncAAM (Version 1.8): Bayesian age-at-maturity analysis from increment-width growth series
#'
#' Fits a Bayesian hierarchical nonlinear mixed-effects model to annual increment-width
#' data and estimates age at maturity using posterior threshold-ratio (TR) and jerk
#' approaches. The model uses a negative exponential curve with fish-level varying
#' parameters.
#'
#' The input data should represent one analysis group. `IncAAM()` does not perform
#' grouping by species, sex, axis, stock, or other variables; users should filter the
#' data before calling the function.
#'
#' @param data A data frame containing annual increment-width measurements, with one row per
#'   increment width. Data should be filtered to the desired analysis group before calling
#'   `IncAAM()`.
#' @param FishID Character string giving the column name identifying individual fish.
#' @param Inc_num Character string giving the column name containing increment number
#'   (used as the age-like predictor).
#' @param Inc_width Character string giving the column name containing increment width.
#' @param sex Character string selecting the sex-specific TR threshold: `"female"`, `"male"`,
#'   or `"custom"`. When `"custom"` is selected, the value supplied to `custom_threshold`
#'   is used.
#' @param female_threshold Numeric TR threshold used when `sex = "female"`.
#'   Default is 0.14.
#' @param male_threshold Numeric TR threshold used when `sex = "male"`.
#'   Default is 0.24.
#' @param custom_threshold Numeric TR threshold used when `sex = "custom"`.
#' @param known_A50 Optional numeric value giving a known or published A50 to display
#'   as a reference in plots.
#' @param min_Inc_num Minimum increment number retained for analysis. Default is 2.
#' @param min_increments Minimum number of increment observations required per fish.
#'   Default is 3.
#' @param min_fish Minimum number of fish required for the hierarchical model. Default
#'   is 3.
#' @param jerk_step Step size used when numerically locating the jerk maximum. Default
#'   is 0.01.
#' @param chains Number of MCMC chains. Default is 4.
#' @param cores Number of CPU cores used for model fitting. Default is 4.
#' @param n_iter Total number of MCMC iterations per chain. Default is 1500.
#' @param n_warmup Number of warm-up iterations per chain. Default is 750.
#' @param n_keepdraws Number of posterior draws retained for downstream summaries.
#' @param seed Random-number seed. Default is 123.
#' @param backend Backend passed to `brms::brm()`. Default is `"cmdstanr"`.
#' @param n_adapt_delta Target acceptance probability for Stan adaptation. Default is
#'   0.995.
#' @param n_logc_sd Prior standard deviation for the fish-level random effect of
#'   `logc`. Default is 0.1.
#' @param max_treedepth Maximum NUTS tree depth. Default is 15.
#' @param make_plots Logical; if `TRUE`, create the posterior distribution and
#'   threshold-crossing ogive plots. Default is `TRUE`.
#' @param save_plots Logical; if `TRUE`, save generated plots as PDF files. Default is
#'   `FALSE`.
#' @param output_dir Directory used when `save_plots = TRUE`. Must be supplied
#'   by the user when plots are to be saved. Default is `NULL`.
#'
#' @details
#' The model describes increment width using a negative exponential function,
#' \deqn{Increment = a + b exp(-c Inc\_num),}
#' with fish-level varying parameters `a`, `b`, and `c`. The parameters are estimated
#' in log space in a Bayesian hierarchical model using `brms`.
#'
#' The TR estimate is derived from the posterior distribution of `c` and the selected
#' threshold. Jerk is calculated numerically from posterior fitted curves. The function
#' then summarizes the posterior estimates across fish and constructs a posterior
#' threshold-crossing ogive. A logistic ogive is fitted to the crossing probabilities,
#' providing an estimated ogive A50 where the fitted probability reaches 0.5.
#'
#' The threshold-crossing ogive describes the probability that the Tangent Ratio (TR)
#' has fallen below the defined TR threshold at a given increment number, and therefore
#' indicates sexual maturity according to the selected threshold. `Ogive_A50` is the
#' estimated increment number at 50% maturity from the fitted logistic ogive. The TR
#' index is the estimated age at 50% maturity when the selected TR threshold is the
#' maturity criterion calibrated for that purpose. The Jerk index is the increment
#' number corresponding to the maximum jerk and is not currently assigned to a defined
#' proportion or stage of sexual maturity.
#'
#' Details of the model, method, outputs and applications are provided in Campana,
#' Smoliński, Morrongiello and Black (2026, in press). Users should consult this
#' publication for the methodological background and interpretation of IncAAM outputs.
#'
#' The function returns the fitted Bayesian model together with fish-level estimates,
#' increment-level threshold-crossing probabilities, posterior group distributions,
#' ogive results, fitted curves, diagnostics, settings, and optional plots.
#'
#' @return An object of class `IncAAM` (a list) containing:
#' \describe{
#'   \item{summary}{Group-level summary of TR and Jerk estimates.}
#'   \item{fish}{Fish-level posterior summaries of model parameters, TR, and Jerk.}
#'   \item{increment}{Fish- and increment-level posterior threshold-crossing probabilities.}
#'   \item{posterior}{Posterior group-level draws for TR and Jerk.}
#'   \item{crossing_ogive}{Observed posterior crossing-ogive summary by increment number.}
#'   \item{crossing_ogive_prediction}{Predicted values from the fitted logistic ogive.}
#'   \item{ogive_summary}{Estimated ogive A50, slope, and intercept.}
#'   \item{ogive_plot}{Threshold-crossing ogive plot, when plots are enabled.}
#'   \item{fitted_curves}{Fish-level fitted negative-exponential curves.}
#'   \item{model}{The fitted `brms` model object.}
#'   \item{diagnostics}{MCMC and sampling diagnostics.}
#'   \item{settings}{Analysis settings used by the function.}
#'   \item{data}{Basic information on the analyzed data.}
#'   \item{plot}{Posterior TR/Jerk density plot, when plots are enabled.}
#' }
#'
#' @examples
#' \donttest{
#' # Example based on the current IncAAM test workflow:
#' library(IncAAM)
#' library(dplyr)
#' 
#' data(plaice)
#' head(plaice)
#' plaice_female <- plaice |>
#' dplyr::filter(Sex == "female")
#'
#' fit <- IncAAM(
#'   data = plaice_female,
#'   FishID = "FishID",
#'   Inc_num = "Inc_num",
#'   Inc_width = "Inc_width_mm",
#'   sex = "female",
#'   known_A50 = 7.6,
#'   n_iter = 1500,
#'   n_warmup = 750,
#'   n_keepdraws = 500,
#'   n_adapt_delta = 0.995,
#'   n_logc_sd = 0.1
#' )
#'
#' #Users can display the complete set of model outputs with `print(fit)`,
#' #or inspect individual output components directly (e.g. `fit$summary`).
#'
#' print(fit)
#'
#' fit$summary
#' fit$crossing_ogive
#' fit$ogive_summary
#' }
#'
#' @export
#' @importFrom stats gaussian coef glm predict quantile median sd var quasibinomial
#' @importFrom utils head tail

IncAAM <- function(
    data,
    
    # --------------------------------------------------------
    # Data columns
    # --------------------------------------------------------
    FishID = "FishID",
    Inc_num = "Inc_num",
    Inc_width = "Inc_width_mm",
    
    # --------------------------------------------------------
    # TR threshold
    # --------------------------------------------------------
    sex = "female",
    female_threshold = 0.14,
    male_threshold = 0.24,
    custom_threshold = NULL,
    
    # --------------------------------------------------------
    # Known / published A50
    # --------------------------------------------------------
    known_A50 = NULL,
    
    # --------------------------------------------------------
    # Data filtering
    # --------------------------------------------------------
    min_Inc_num = 2,
    min_increments = 3,
    min_fish = 3,
    
    # --------------------------------------------------------
    # Jerk calculation
    # --------------------------------------------------------
    jerk_step = 0.01,
    
    # --------------------------------------------------------
    # Bayesian model
    # --------------------------------------------------------
    chains = 4,
    cores = 4,
    n_iter = 1500,
    n_warmup = 750,
    n_keepdraws = 500,
    seed = 123,
    backend = "cmdstanr",
    n_adapt_delta = 0.995,
    n_logc_sd = 0.1,
    max_treedepth = 15,
    
    # --------------------------------------------------------
    # Output / plots
    # --------------------------------------------------------
    make_plots = TRUE,
    save_plots = FALSE,
    output_dir = NULL
) {
  
  # ==========================================================
  # 2. CHECK PACKAGES
  # ==========================================================
  
  required_packages <- c(
    "dplyr",
    "tidyr",
    "purrr",
    "tibble",
    "ggplot2",
    "brms",
    "posterior",
    "minpack.lm"
  )
  
  missing_packages <- required_packages[
    !vapply(
      required_packages,
      requireNamespace,
      logical(1),
      quietly = TRUE
    )
  ]
  
  if (length(missing_packages) > 0) {
    stop(
      "The following packages are required: ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 3. CHECK INPUT
  # ==========================================================
  
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.", call. = FALSE)
  }
  
  required_columns <- c(
    FishID,
    Inc_num,
    Inc_width
  )
  
  missing_columns <- setdiff(
    required_columns,
    names(data)
  )
  
  if (length(missing_columns) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 4. CHECK SEX / TR THRESHOLD
  # ==========================================================
  
  sex <- match.arg(
    sex,
    choices = c("female", "male", "custom")
  )
  
  if (sex == "female") {
    
    TR_threshold <- female_threshold
    
  } else if (sex == "male") {
    
    TR_threshold <- male_threshold
    
  } else {
    
    if (is.null(custom_threshold) ||
        length(custom_threshold) != 1 ||
        !is.finite(custom_threshold)) {
      
      stop(
        "When sex = 'custom', ",
        "custom_threshold must be a single finite numeric value.",
        call. = FALSE
      )
    }
    
    TR_threshold <- custom_threshold
  }
  
  if (!is.finite(TR_threshold) || TR_threshold <= 0) {
    stop(
      "TR threshold must be a positive finite number.",
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 5. CHECK MODEL SETTINGS
  # ==========================================================
  
  if (n_iter <= n_warmup) {
    stop(
      "'n_iter' must be greater than 'n_warmup'.",
      call. = FALSE
    )
  }
  
  if (min_fish < 3) {
    stop(
      "'min_fish' must be at least 3.",
      call. = FALSE
    )
  }
  
  if (min_increments < 1) {
    stop(
      "'min_increments' must be at least 1.",
      call. = FALSE
    )
  }
  
  if (jerk_step <= 0) {
    stop(
      "'jerk_step' must be > 0.",
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 6. PREPARE DATA
  #
  # ==========================================================
  
  dat <- data |>
    
    dplyr::rename(
      FishID_actual = dplyr::all_of(FishID),
      Inc_num_actual = dplyr::all_of(Inc_num),
      Increment = dplyr::all_of(Inc_width)
    ) |>
    
    dplyr::mutate(
      FishID_actual = as.character(FishID_actual),
      Inc_num = suppressWarnings(
        as.numeric(Inc_num_actual)
      ),
      Increment = suppressWarnings(
        as.numeric(Increment)
      )
    ) |>
    
    dplyr::filter(
      is.finite(Inc_num),
      is.finite(Increment)
    ) |>
    
    dplyr::filter(
      Inc_num >= min_Inc_num
    ) |>
    
    dplyr::arrange(
      FishID_actual,
      Inc_num
    )
  
  
  # ==========================================================
  # 7. CHECK DATA AFTER BASIC FILTERING
  # ==========================================================
  
  if (nrow(dat) == 0) {
    stop(
      "No observations remain after filtering.",
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 8. REMOVE FISH WITH TOO FEW INCREMENTS
  # ==========================================================
  
  fish_counts <- dat |>
    dplyr::count(
      FishID_actual,
      name = "n_obs"
    )
  
  keep_fish <- fish_counts |>
    dplyr::filter(
      n_obs >= min_increments
    ) |>
    dplyr::pull(
      FishID_actual
    )
  
  dat <- dat |>
    dplyr::filter(
      FishID_actual %in% keep_fish
    )
  
  
  # ==========================================================
  # 9. CHECK NUMBER OF FISH
  # ==========================================================
  
  n_fish <- dplyr::n_distinct(
    dat$FishID_actual
  )
  
  if (n_fish < min_fish) {
    
    stop(
      "Too few fish for hierarchical model: ",
      n_fish,
      " fish available, but min_fish = ",
      min_fish,
      ".",
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 10. FISH METADATA
  # ==========================================================
  
  fish_meta <- dat |>
    
    dplyr::group_by(
      FishID_actual
    ) |>
    
    dplyr::summarise(
      min_inc = min(
        Inc_num,
        na.rm = TRUE
      ),
      
      max_inc = max(
        Inc_num,
        na.rm = TRUE
      ),
      
      n_obs = dplyr::n(),
      
      .groups = "drop"
    )
  
  
  # ==========================================================
  # 11. POOLED NLS STARTING VALUES
  # ==========================================================
  
  pooled <- dat |>
    
    dplyr::group_by(
      Inc_num
    ) |>
    
    dplyr::summarise(
      meanIncrement = mean(
        Increment,
        na.rm = TRUE
      ),
      .groups = "drop"
    )
  
  
  start_values <- list(
    a = 0.01,
    b = 0.3,
    c = 0.5
  )
  
  
  fit_nls <- try(
    
    minpack.lm::nlsLM(
      
      meanIncrement ~
        a +
        b * exp(-c * Inc_num),
      
      data = pooled,
      
      start = list(
        a = max(
          min(
            pooled$meanIncrement,
            na.rm = TRUE
          ),
          1e-4
        ),
        
        b = max(
          diff(
            range(
              pooled$meanIncrement,
              na.rm = TRUE
            )
          ),
          1e-4
        ),
        
        c = 0.3
      ),
      
      lower = c(
        a = 1e-8,
        b = 1e-8,
        c = 1e-8
      ),
      
      upper = c(
        a = Inf,
        b = Inf,
        c = Inf
      ),
      
      control =
        minpack.lm::nls.lm.control(
          maxiter = 500
        )
    ),
    
    silent = TRUE
  )
  
  
  if (!inherits(fit_nls, "try-error")) {
    
    cf <- coef(fit_nls)
    
    start_values <- list(
      a = unname(cf["a"]),
      b = unname(cf["b"]),
      c = unname(cf["c"])
    )
  }
  
  
  # ==========================================================
  # 12. LOG-SCALE STARTING VALUES
  # ==========================================================
  
  loga0 <- log(
    max(
      start_values$a,
      1e-6
    )
  )
  
  logb0 <- log(
    max(
      start_values$b,
      1e-6
    )
  )
  
  logc0 <- log(
    max(
      start_values$c,
      1e-6
    )
  )
  
  
  # ==========================================================
  # 13. DATA FOR BRMS
  # ==========================================================
  
  dat_model <- dat |>
    
    dplyr::mutate(
      FishID_actual =
        factor(FishID_actual)
    )
  
  
  # ==========================================================
  # 14. BAYESIAN MODEL
  # ==========================================================
  
  bform <- brms::bf(
    
    Increment ~
      exp(loga) +
      exp(logb) *
      exp(
        -exp(logc) * Inc_num
      ),
    
    loga ~
      1 +
      (1 | FishID_actual),
    
    logb ~
      1 +
      (1 | FishID_actual),
    
    logc ~
      1 +
      (1 | FishID_actual),
    
    nl = TRUE
  )
  
  
  # ==========================================================
  # 15. PRIORS
  # ==========================================================
  
  priors <- c(
    
    brms::set_prior(
      paste0(
        "normal(",
        loga0,
        ", 1.0)"
      ),
      nlpar = "loga"
    ),
    
    brms::set_prior(
      paste0(
        "normal(",
        logb0,
        ", 1.0)"
      ),
      nlpar = "logb"
    ),
    
    brms::set_prior(
      paste0(
        "normal(",
        logc0,
        ", 1.0)"
      ),
      nlpar = "logc"
    ),
    
    brms::set_prior(
      "normal(0, 0.5)",
      class = "sd",
      group = "FishID_actual",
      nlpar = "loga"
    ),
    
    brms::set_prior(
      "normal(0, 0.5)",
      class = "sd",
      group = "FishID_actual",
      nlpar = "logb"
    ),
    
    brms::set_prior(
      paste0(
        "normal(0, ",
        n_logc_sd,
        ")"
      ),
      class = "sd",
      group = "FishID_actual",
      nlpar = "logc"
    ),
    
    brms::set_prior(
      "exponential(1)",
      class = "sigma"
    )
  )
  
  
  # ==========================================================
  # 16. FIT MODEL
  # ==========================================================
  
  fit <- brms::brm(
    
    formula = bform,
    
    data = dat_model,
    
    family = gaussian(),
    
    prior = priors,
    
    chains = chains,
    
    cores = cores,
    
    iter = n_iter,
    
    warmup = n_warmup,
    
    seed = seed,
    
    backend = backend,
    
    control = list(
      adapt_delta = n_adapt_delta,
      max_treedepth = max_treedepth
    ),
    
    refresh = 1000
  )
  
  
  # ==========================================================
  # 17. EXTRACT FISH-LEVEL POSTERIOR COEFFICIENTS
  # ==========================================================
  
  coef_draws <-
    coef(
      fit,
      summary = FALSE
    )$FishID_actual
  
  fish_levels <-
    as.character(
      dimnames(coef_draws)[[2]]
    )
  
  coef_names <-
    dimnames(coef_draws)[[3]]
  
  
  a_name <-
    coef_names[
      grepl(
        "^loga_",
        coef_names
      )
    ]
  
  b_name <-
    coef_names[
      grepl(
        "^logb_",
        coef_names
      )
    ]
  
  c_name <-
    coef_names[
      grepl(
        "^logc_",
        coef_names
      )
    ]
  
  
  if (
    length(a_name) != 1 ||
    length(b_name) != 1 ||
    length(c_name) != 1
  ) {
    
    stop(
      "Could not identify fish-level posterior coefficients.",
      call. = FALSE
    )
  }
  
  
  # ==========================================================
  # 18. SELECT POSTERIOR DRAWS
  # ==========================================================
  
  n_draws_total <-
    dim(coef_draws)[1]
  
  if (
    !is.null(n_keepdraws) &&
    n_keepdraws < n_draws_total
  ) {
    
    draw_idx <-
      sort(
        unique(
          round(
            seq(
              1,
              n_draws_total,
              length.out = n_keepdraws
            )
          )
        )
      )
    
  } else {
    
    draw_idx <-
      seq_len(n_draws_total)
  }
  
  
  # ==========================================================
  # 19. HELPER: TR FROM c
  # ==========================================================
  
  compute_TR <- function(
    c_draw,
    threshold,
    max_inc
  ) {
    
    x_tr <-
      2 -
      log(threshold) /
      c_draw
    
    ifelse(
      is.finite(x_tr) &
        x_tr <= max_inc,
      x_tr,
      NA_real_
    )
  }
  
  
  # ==========================================================
  # 20. HELPER: JERK
  # ==========================================================
  
  compute_Jerk <- function(
    a,
    b,
    c,
    min_inc,
    max_inc,
    step
  ) {
    
    x <-
      seq(
        min_inc,
        max_inc,
        by = step
      )
    
    fitted <-
      a +
      b *
      exp(
        -c * x
      )
    
    rel_diff <-
      fitted /
      dplyr::lag(fitted)
    
    jerk <-
      rel_diff -
      dplyr::lag(rel_diff)
    
    local_max <-
      which(
        dplyr::lag(jerk, 1) < jerk &
          dplyr::lead(jerk, 1) < jerk
      )
    
    if (length(local_max) == 1) {
      
      return(
        x[local_max - 1]
      )
      
    } else {
      
      return(NA_real_)
    }
  }
  
  
  # ==========================================================
  # 21. FISH-LEVEL POSTERIOR SUMMARY
  # ==========================================================
  
  fish_posterior_summary <- purrr::map_dfr(
    
    fish_levels,
    
    function(fid) {
      
      fish_row <-
        fish_meta |>
        dplyr::filter(
          FishID_actual == fid
        )
      
      min_inc_i <-
        fish_row$min_inc[[1]]
      
      max_inc_i <-
        fish_row$max_inc[[1]]
      
      
      loga_draw <-
        coef_draws[
          draw_idx,
          fid,
          a_name
        ]
      
      logb_draw <-
        coef_draws[
          draw_idx,
          fid,
          b_name
        ]
      
      logc_draw <-
        coef_draws[
          draw_idx,
          fid,
          c_name
        ]
      
      
      a_draw <- exp(loga_draw)
      b_draw <- exp(logb_draw)
      c_draw <- exp(logc_draw)
      
      
      # ------------------------------------------------------
      # TR
      # ------------------------------------------------------
      
      TR_draw <-
        compute_TR(
          c_draw = c_draw,
          threshold = TR_threshold,
          max_inc = max_inc_i
        )
      
      
      # ------------------------------------------------------
      # Jerk
      # ------------------------------------------------------
      
      Jerk_draw <-
        purrr::map_dbl(
          
          seq_along(a_draw),
          
          function(ii) {
            
            compute_Jerk(
              a = a_draw[ii],
              b = b_draw[ii],
              c = c_draw[ii],
              min_inc = min_inc_i,
              max_inc = max_inc_i,
              step = jerk_step
            )
          }
        )
      
      
      tibble::tibble(
        
        FishID = fid,
        
        min_inc = min_inc_i,
        
        max_inc = max_inc_i,
        
        n_obs = fish_row$n_obs[[1]],
        
        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------
        
        a_median =
          median(
            a_draw,
            na.rm = TRUE
          ),
        
        a_q05 =
          quantile(
            a_draw,
            0.05,
            na.rm = TRUE
          ),
        
        a_q95 =
          quantile(
            a_draw,
            0.95,
            na.rm = TRUE
          ),
        
        b_median =
          median(
            b_draw,
            na.rm = TRUE
          ),
        
        b_q05 =
          quantile(
            b_draw,
            0.05,
            na.rm = TRUE
          ),
        
        b_q95 =
          quantile(
            b_draw,
            0.95,
            na.rm = TRUE
          ),
        
        c_median =
          median(
            c_draw,
            na.rm = TRUE
          ),
        
        c_q05 =
          quantile(
            c_draw,
            0.05,
            na.rm = TRUE
          ),
        
        c_q95 =
          quantile(
            c_draw,
            0.95,
            na.rm = TRUE
          ),
        
        # ----------------------------------------------------
        # TR
        # ----------------------------------------------------
        
        TR_median =
          if (
            all(is.na(TR_draw))
          ) NA_real_
        else median(
          TR_draw,
          na.rm = TRUE
        ),
        
        TR_mean =
          if (
            all(is.na(TR_draw))
          ) NA_real_
        else mean(
          TR_draw,
          na.rm = TRUE
        ),
        
        TR_q05 =
          if (
            all(is.na(TR_draw))
          ) NA_real_
        else quantile(
          TR_draw,
          0.05,
          na.rm = TRUE
        ),
        
        TR_q95 =
          if (
            all(is.na(TR_draw))
          ) NA_real_
        else quantile(
          TR_draw,
          0.95,
          na.rm = TRUE
        ),
        
        TR_reached_prob =
          mean(
            is.finite(TR_draw)
          ),
        
        # ----------------------------------------------------
        # Jerk
        # ----------------------------------------------------
        
        Jerk_median =
          if (
            all(is.na(Jerk_draw))
          ) NA_real_
        else median(
          Jerk_draw,
          na.rm = TRUE
        ),
        
        Jerk_mean =
          if (
            all(is.na(Jerk_draw))
          ) NA_real_
        else mean(
          Jerk_draw,
          na.rm = TRUE
        ),
        
        Jerk_q05 =
          if (
            all(is.na(Jerk_draw))
          ) NA_real_
        else quantile(
          Jerk_draw,
          0.05,
          na.rm = TRUE
        ),
        
        Jerk_q95 =
          if (
            all(is.na(Jerk_draw))
          ) NA_real_
        else quantile(
          Jerk_draw,
          0.95,
          na.rm = TRUE
        )
      )
    }
  )
  
  
  # ==========================================================
  # 22. INCREMENT-LEVEL THRESHOLD CROSSING PROBABILITIES
  # ==========================================================
  
  fish_incnum_posterior_summary <-
    purrr::map_dfr(
      
      fish_levels,
      
      function(fid) {
        
        fish_row <-
          fish_meta |>
          dplyr::filter(
            FishID_actual == fid
          )
        
        max_inc_i <-
          fish_row$max_inc[[1]]
        
        logc_draw <-
          coef_draws[
            draw_idx,
            fid,
            c_name
          ]
        
        c_draw <-
          exp(logc_draw)
        
        TR_draw <-
          compute_TR(
            c_draw = c_draw,
            threshold = TR_threshold,
            max_inc = max_inc_i
          )
        
        fish_incnums <-
          dat |>
          dplyr::filter(
            FishID_actual == fid
          ) |>
          dplyr::distinct(
            Inc_num
          ) |>
          dplyr::arrange(
            Inc_num
          ) |>
          dplyr::pull(
            Inc_num
          )
        
        
        purrr::map_dfr(
          
          fish_incnums,
          
          function(xi) {
            
            prob_crossed <-
              mean(
                ifelse(
                  is.finite(TR_draw),
                  TR_draw <= xi,
                  FALSE
                ),
                na.rm = TRUE
              )
            
            tibble::tibble(
              
              FishID = fid,
              
              Inc_num = xi,
              
              prob_crossed_by_inc =
                prob_crossed,
              
              prob_not_crossed_by_inc =
                1 - prob_crossed,
              
              n_draws_used =
                length(TR_draw)
            )
          }
        )
      }
    )
  
  
  # ==========================================================
  # 23. GROUP-LEVEL SUMMARY
  #
  # "Group" here means simply the entire input data set.
  # There is no Species/Sex/Axis grouping.
  # ==========================================================
  
  TR_values <-
    fish_posterior_summary$TR_median
  
  Jerk_values <-
    fish_posterior_summary$Jerk_median
  
  
  summary_table <- tibble::tibble(
    
    Index = c(
      "TR",
      "Jerk"
    ),
    
    threshold =
      c(
        TR_threshold,
        NA_real_
      ),
    
    mean = c(
      
      mean(
        TR_values,
        na.rm = TRUE
      ),
      
      mean(
        Jerk_values,
        na.rm = TRUE
      )
    ),
    
    median = c(
      
      median(
        TR_values,
        na.rm = TRUE
      ),
      
      median(
        Jerk_values,
        na.rm = TRUE
      )
    ),
    
    se = c(
      
      sd(
        TR_values,
        na.rm = TRUE
      ) /
        sqrt(
          sum(
            is.finite(
              TR_values
            )
          )
        ),
      
      sd(
        Jerk_values,
        na.rm = TRUE
      ) /
        sqrt(
          sum(
            is.finite(
              Jerk_values
            )
          )
        )
    ),
    
    p5 = c(
      
      quantile(
        TR_values,
        0.05,
        na.rm = TRUE
      ),
      
      quantile(
        Jerk_values,
        0.05,
        na.rm = TRUE
      )
    ),
    
    p95 = c(
      
      quantile(
        TR_values,
        0.95,
        na.rm = TRUE
      ),
      
      quantile(
        Jerk_values,
        0.95,
        na.rm = TRUE
      )
    ),
    
    sample_n = c(
      
      sum(
        is.finite(
          TR_values
        )
      ),
      
      sum(
        is.finite(
          Jerk_values
        )
      )
    ),
    
    proportion_success = c(
      
      mean(
        is.finite(
          TR_values
        )
      ),
      
      mean(
        is.finite(
          Jerk_values
        )
      )
    )
  )
  
  
  # ==========================================================
  # 24. POSTERIOR GROUP DRAW DISTRIBUTIONS
  #
  # Each posterior draw is the mean across fish.
  # ==========================================================
  
  tr_mat <-
    matrix(
      NA_real_,
      nrow = length(draw_idx),
      ncol = length(fish_levels)
    )
  
  jerk_mat <-
    matrix(
      NA_real_,
      nrow = length(draw_idx),
      ncol = length(fish_levels)
    )
  
  
  for (jj in seq_along(fish_levels)) {
    
    fid <-
      fish_levels[jj]
    
    fish_row <-
      fish_meta |>
      dplyr::filter(
        FishID_actual == fid
      )
    
    min_inc_i <-
      fish_row$min_inc[[1]]
    
    max_inc_i <-
      fish_row$max_inc[[1]]
    
    
    loga_draw <-
      coef_draws[
        draw_idx,
        fid,
        a_name
      ]
    
    logb_draw <-
      coef_draws[
        draw_idx,
        fid,
        b_name
      ]
    
    logc_draw <-
      coef_draws[
        draw_idx,
        fid,
        c_name
      ]
    
    
    a_draw <- exp(loga_draw)
    b_draw <- exp(logb_draw)
    c_draw <- exp(logc_draw)
    
    
    tr_mat[, jj] <-
      compute_TR(
        c_draw,
        TR_threshold,
        max_inc_i
      )
    
    
    jerk_mat[, jj] <-
      purrr::map_dbl(
        
        seq_along(a_draw),
        
        function(ii) {
          
          compute_Jerk(
            a = a_draw[ii],
            b = b_draw[ii],
            c = c_draw[ii],
            min_inc = min_inc_i,
            max_inc = max_inc_i,
            step = jerk_step
          )
        }
      )
  }
  
  
  group_TR_draw <-
    apply(
      tr_mat,
      1,
      function(x)
        mean(
          x,
          na.rm = TRUE
        )
    )
  
  
  group_Jerk_draw <-
    apply(
      jerk_mat,
      1,
      function(x)
        mean(
          x,
          na.rm = TRUE
        )
    )
  
  
  posterior_draws <- dplyr::bind_rows(
    
    tibble::tibble(
      Index = "TR",
      draw = seq_along(
        group_TR_draw
      ),
      value = group_TR_draw
    ),
    
    tibble::tibble(
      Index = "Jerk",
      draw = seq_along(
        group_Jerk_draw
      ),
      value = group_Jerk_draw
    )
  ) |>
    dplyr::filter(
      is.finite(value)
    )
  
  
  # ==========================================================
  # 25. POSTERIOR CROSSING OGIVE
  # ==========================================================
  
  crossing_ogive <-
    fish_incnum_posterior_summary |>
    
    dplyr::group_by(
      Inc_num
    ) |>
    
    dplyr::summarise(
      
      mean_prob_crossed =
        mean(
          prob_crossed_by_inc,
          na.rm = TRUE
        ),
      
      sd_prob_crossed =
        sd(
          prob_crossed_by_inc,
          na.rm = TRUE
        ),
      
      se_prob_crossed =
        sd_prob_crossed /
        sqrt(
          dplyr::n()
        ),
      
      n_fish =
        dplyr::n(),
      
      .groups = "drop"
    ) |>
    
    dplyr::arrange(
      Inc_num
    )
  
  
  # Give the crossing-ogive table a dedicated print method so that
  # all increment rows are shown by default in the console.
  class(crossing_ogive) <-
    c("IncAAM_crossing_ogive", class(crossing_ogive))


  # ==========================================================
  # 26. FIT LOGISTIC OGIVE
  # ==========================================================
  
  crossing_ogive_prediction <-
    tibble::tibble()
  
  
  if (
    nrow(crossing_ogive) >= 3 &&
    length(
      unique(
        crossing_ogive$Inc_num
      )
    ) >= 3
  ) {
    
    ogive_data <-
      crossing_ogive |>
      dplyr::mutate(
        
        mean_prob_clamped =
          pmin(
            pmax(
              mean_prob_crossed,
              1e-4
            ),
            1 - 1e-4
          )
      )
    
    
    ogive_model <-
      try(
        
        stats::glm(
          
          mean_prob_clamped ~
            Inc_num,
          
          weights = n_fish,
          
          family =
            quasibinomial(
              link = "logit"
            ),
          
          data = ogive_data
        ),
        
        silent = TRUE
      )
    
    
    if (
      !inherits(
        ogive_model,
        "try-error"
      )
    ) {
      
      pred_grid <-
        tibble::tibble(
          
          Inc_num =
            seq(
              min(
                crossing_ogive$Inc_num
              ),
              max(
                crossing_ogive$Inc_num
              ),
              by = 0.01
            )
        )
      
      
      pred_grid$pred_prob <-
        stats::predict(
          ogive_model,
          newdata = pred_grid,
          type = "response"
        )
      
      
      crossing_ogive_prediction <-
        pred_grid
    }
    
  } else {
    
    ogive_model <- NULL
  }
  
  # ==========================================================
  # 26b. OGIVE SUMMARY
  # ==========================================================
  
  ogive_summary <- tibble::tibble(
    
    Ogive_A50 =
      NA_real_,
    
    Ogive_slope =
      NA_real_,
    
    Ogive_intercept =
      NA_real_
  )
  
  
  if (
    !is.null(ogive_model) &&
    inherits(ogive_model, "glm")
  ) {
    
    # --------------------------------------------------------
    # A50 = increment number at predicted probability = 0.5
    # --------------------------------------------------------
    
    beta_0 <-
      stats::coef(
        ogive_model
      )[1]
    
    beta_1 <-
      stats::coef(
        ogive_model
      )[2]
    
    
    ogive_A50 <-
      -beta_0 / beta_1
    
    
    ogive_summary <-
      tibble::tibble(
        
        Ogive_A50 =
          ogive_A50,
        
        Ogive_slope =
          beta_1,
        
        Ogive_intercept =
          beta_0
      )
  }
  
  # ==========================================================
  # 26c. CROSSING OGIVE PLOT
  # ==========================================================
  
  crossing_ogive_plot <- NULL
  
  if (
    isTRUE(make_plots) &&
    nrow(crossing_ogive) > 0
  ) {
    
    crossing_ogive_plot <-
      ggplot2::ggplot(
        crossing_ogive,
        ggplot2::aes(
          x = Inc_num,
          y = mean_prob_crossed
        )
      ) +
      
      ggplot2::geom_point(
        size = 2.5
      ) +
      
      ggplot2::geom_errorbar(
        ggplot2::aes(
          ymin = pmax(
            mean_prob_crossed -
              se_prob_crossed,
            0
          ),
          ymax = pmin(
            mean_prob_crossed +
              se_prob_crossed,
            1
          )
        ),
        width = 0
      )
    
    
    # --------------------------------------------------------
    # Fitted logistic ogive
    # --------------------------------------------------------
    
    if (
      nrow(crossing_ogive_prediction) > 0
    ) {
      
      crossing_ogive_plot <-
        crossing_ogive_plot +
        
        ggplot2::geom_line(
          data =
            crossing_ogive_prediction,
          ggplot2::aes(
            x = Inc_num,
            y = pred_prob
          ),
          linewidth = 1
        )
    }
    
    
    # --------------------------------------------------------
    # Expected A50
    # --------------------------------------------------------
    
    if (!is.null(known_A50)) {
      
      crossing_ogive_plot <-
        crossing_ogive_plot +
        
        ggplot2::geom_vline(
          xintercept =
            known_A50,
          linetype = 2,
          linewidth = 0.8,
          colour = "red"
        )
    }
    
    
    # --------------------------------------------------------
    # Estimated ogive A50
    # --------------------------------------------------------
    
    if (
      is.finite(
        ogive_summary$Ogive_A50[1]
      )
    ) {
      
      crossing_ogive_plot <-
        crossing_ogive_plot +
        
        ggplot2::geom_vline(
          xintercept =
            ogive_summary$Ogive_A50[1],
          linetype = 2,
          linewidth = 0.8,
          colour = "blue"
        )
    }
    
    
    crossing_ogive_plot <-
      crossing_ogive_plot +
      
      ggplot2::coord_cartesian(
        ylim = c(0, 1)
      ) +
      
      ggplot2::labs(
        
        title =
          "IncAAM posterior threshold-crossing ogive",
        
        subtitle =
          paste0(
            "TR threshold = ",
            TR_threshold,
            
            if (!is.null(known_A50))
              paste0(
                " | Expected A50 = ",
                known_A50
              )
            else
              "",
            
            if (
              is.finite(
                ogive_summary$Ogive_A50[1]
              )
            )
              paste0(
                " | Ogive A50 = ",
                round(
                  ogive_summary$Ogive_A50[1],
                  2
                )
              )
            else
              ""
          ),
        
        x =
          "Increment number",
        
        y =
          "Probability TR is below threshold (sexual maturity)"
      ) +
      
      ggplot2::theme_classic(
        base_size = 14
      )
  }
  # ==========================================================
  # 27. DIAGNOSTICS
  # ==========================================================
  
  draw_array <-
    posterior::as_draws_array(
      fit
    )
  
  draw_summary <-
    posterior::summarise_draws(
      draw_array,
      "rhat",
      "ess_bulk",
      "ess_tail"
    )
  
  
  nuts <-
    brms::nuts_params(
      fit
    )
  
  
  n_divergent <-
    nuts |>
    
    dplyr::filter(
      Parameter == "divergent__"
    ) |>
    
    dplyr::summarise(
      n = sum(
        Value,
        na.rm = TRUE
      )
    ) |>
    
    dplyr::pull(n)
  
  
  n_max_tree_depth <-
    nuts |>
    
    dplyr::filter(
      Parameter == "treedepth__"
    ) |>
    
    dplyr::summarise(
      n = sum(
        Value >= max_treedepth,
        na.rm = TRUE
      )
    ) |>
    
    dplyr::pull(n)
  
  
  bfmi_values <-
    nuts |>
    
    dplyr::filter(
      Parameter == "energy__"
    ) |>
    
    dplyr::group_by(
      Chain
    ) |>
    
    dplyr::summarise(
      
      bfmi =
        mean(
          diff(Value)^2,
          na.rm = TRUE
        ) /
        var(
          Value,
          na.rm = TRUE
        ),
      
      .groups = "drop"
    ) |>
    
    dplyr::pull(
      bfmi
    )
  
  
  diagnostics <- tibble::tibble(
    
    n_fish =
      n_fish,
    
    n_observations =
      nrow(dat),
    
    n_draws_post_warmup =
      posterior::ndraws(
        draw_array
      ),
    
    max_rhat =
      max(
        draw_summary$rhat,
        na.rm = TRUE
      ),
    
    min_bulk_ESS =
      min(
        draw_summary$ess_bulk,
        na.rm = TRUE
      ),
    
    min_tail_ESS =
      min(
        draw_summary$ess_tail,
        na.rm = TRUE
      ),
    
    n_divergent =
      ifelse(
        length(n_divergent) == 0,
        NA_integer_,
        n_divergent
      ),
    
    n_max_tree_depth =
      ifelse(
        length(n_max_tree_depth) == 0,
        NA_integer_,
        n_max_tree_depth
      ),
    
    mean_BFMI =
      if (
        length(bfmi_values) == 0
      )
        NA_real_
    else
      mean(
        bfmi_values,
        na.rm = TRUE
      )
  )
  
  
  # ==========================================================
  # 28. FITTED CURVES
  # ==========================================================
  
  fitted_curves <-
    purrr::map_dfr(
      
      fish_levels,
      
      function(fid) {
        
        fish_row <-
          fish_meta |>
          dplyr::filter(
            FishID_actual == fid
          )
        
        min_inc_i <-
          fish_row$min_inc[[1]]
        
        max_inc_i <-
          fish_row$max_inc[[1]]
        
        
        loga_draw <-
          coef_draws[
            draw_idx,
            fid,
            a_name
          ]
        
        logb_draw <-
          coef_draws[
            draw_idx,
            fid,
            b_name
          ]
        
        logc_draw <-
          coef_draws[
            draw_idx,
            fid,
            c_name
          ]
        
        
        x <-
          seq(
            min_inc_i,
            max_inc_i,
            by = 0.01
          )
        
        
        tibble::tibble(
          
          FishID = fid,
          
          Inc_num = x,
          
          fitted =
            median(
              exp(loga_draw),
              na.rm = TRUE
            ) +
            
            median(
              exp(logb_draw),
              na.rm = TRUE
            ) *
            
            exp(
              -median(
                exp(logc_draw),
                na.rm = TRUE
              ) *
                x
            )
        )
      }
    )
  
  
  # ==========================================================
  # 29. RESULT PLOT
  # ==========================================================
  
  result_plot <- NULL
  
  if (isTRUE(make_plots)) {
    
    plot_data <-
      posterior_draws |>
      dplyr::filter(
        is.finite(value)
      ) |>
      dplyr::mutate(
        Index = factor(
          Index,
          levels = c("TR", "Jerk")
        )
      )
    
    
    # --------------------------------------------------------
    # Lines: posterior mean and expected A50
    # --------------------------------------------------------
    
    line_data <-
      plot_data |>
      dplyr::group_by(Index) |>
      dplyr::summarise(
        
        posterior_mean =
          mean(
            value,
            na.rm = TRUE
          ),
        
        .groups = "drop"
      ) |>
      tidyr::pivot_longer(
        
        cols = c(
          posterior_mean
        ),
        
        names_to = "line_type",
        
        values_to = "xint"
      ) |>
      dplyr::mutate(
        
        label =
          "Mean",
        
        line_colour =
          "blue"
      )
    
    
    # Add known A50, if supplied
    if (!is.null(known_A50)) {
      
      line_data <-
        dplyr::bind_rows(
          
          line_data,
          
          tibble::tibble(
            
            Index =
              factor(
                c("TR", "Jerk"),
                levels = c("TR", "Jerk")
              ),
            
            line_type =
              "expected_A50",
            
            xint =
              known_A50,
            
            label =
              "A50",
            
            line_colour =
              "red"
          )
        )
    }
    
    
    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    
    result_plot <-
      ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = value
        )
      ) +
      
      ggplot2::geom_density(
        linewidth = 1
      ) +
      
      ggplot2::geom_vline(
        data = line_data,
        ggplot2::aes(
          xintercept = xint,
          colour = line_colour
        ),
        linetype = 2,
        linewidth = 0.8,
        show.legend = FALSE
      ) +
      
      ggplot2::geom_text(
        data = line_data,
        ggplot2::aes(
          x = xint,
          y = Inf,
          label = label,
          colour = line_colour
        ),
        angle = 90,
        vjust = -0.4,
        hjust = 1.1,
        size = 3.5,
        show.legend = FALSE
      ) +
      
      ggplot2::facet_wrap(
        ~Index,
        ncol = 1,
        scales = "fixed"
      ) +
      
      ggplot2::scale_colour_identity() +
      
      ggplot2::labs(
        
        title =
          "IncAAM posterior distributions",
        
        subtitle =
          paste0(
            "TR threshold = ",
            TR_threshold,
            if (!is.null(known_A50))
              paste0(
                " | Expected A50 = ",
                known_A50
              )
            else
              ""
          ),
        
        x =
          "Increment number",
        
        y =
          "Posterior density"
      ) +
      
      ggplot2::theme_classic(
        base_size = 14
      )
  }  
  
  # ==========================================================
  # 30. SAVE PLOTS
  # ==========================================================
  
  if (
    isTRUE(make_plots) &&
    isTRUE(save_plots)
  ) {
    
    if (
      is.null(output_dir) ||
      length(output_dir) != 1L ||
      !is.character(output_dir) ||
      is.na(output_dir) ||
      !nzchar(output_dir)
    ) {
      stop(
        "'output_dir' must be supplied as a non-empty character string ",
        "when save_plots = TRUE.",
        call. = FALSE
      )
    }
    
    if (
      !dir.exists(output_dir)
    ) {
      
      dir.create(
        output_dir,
        recursive = TRUE
      )
    }
    
    
    ggplot2::ggsave(
      
      filename =
        file.path(
          output_dir,
          "IncAAM_posterior_TR_Jerk.pdf"
        ),
      
      plot =
        result_plot,
      
      width = 8,
      
      height = 7
    )
    
    if (!is.null(crossing_ogive_plot)) {
      
      ggplot2::ggsave(
        
        filename =
          file.path(
            output_dir,
            "IncAAM_crossing_ogive.pdf"
          ),
        
        plot =
          crossing_ogive_plot,
        
        width = 7,
        height = 5
      )
    }
  }
  
  
  # ==========================================================
  # 31. RETURN OBJECT
  # ==========================================================
  
  result <- list(
    
    # --------------------------------------------------------
    # Main results
    # --------------------------------------------------------
    
    summary =
      summary_table,
    
    fish =
      fish_posterior_summary,
    
    increment =
      fish_incnum_posterior_summary,
    
    posterior =
      posterior_draws,
    
    crossing_ogive =
      crossing_ogive,
    
    crossing_ogive_prediction =
      crossing_ogive_prediction,
    
    ogive_summary =
      ogive_summary,
    
    ogive_plot =
      crossing_ogive_plot,
    
    fitted_curves =
      fitted_curves,
    
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    
    model =
      fit,
    
    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    
    diagnostics =
      diagnostics,
    
    # --------------------------------------------------------
    # Input information
    # --------------------------------------------------------
    
    settings =
      list(
        
        FishID =
          FishID,
        
        Inc_num =
          Inc_num,
        
        Inc_width =
          Inc_width,
        
        sex =
          sex,
        
        TR_threshold =
          TR_threshold,
        
        female_threshold =
          female_threshold,
        
        male_threshold =
          male_threshold,
        
        custom_threshold =
          custom_threshold,
        
        known_A50 =
          known_A50,
        
        min_Inc_num =
          min_Inc_num,
        
        min_increments =
          min_increments,
        
        min_fish =
          min_fish,
        
        jerk_step =
          jerk_step
      ),
    
    data =
      list(
        
        n_fish =
          n_fish,
        
        n_observations =
          nrow(dat),
        
        fish_meta =
          fish_meta
      ),
    
    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    
    plot =
      result_plot
  )
  
  
  class(result) <-
    c(
      "IncAAM",
      "list"
    )
  
  
  return(result)
}

#' @export
print.IncAAM_crossing_ogive <- function(x, ...) {
  print(
    as.data.frame(x),
    row.names = FALSE
  )
  invisible(x)
}
