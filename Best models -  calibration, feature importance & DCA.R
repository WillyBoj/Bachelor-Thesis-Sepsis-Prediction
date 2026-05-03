# ============================================================
# COMBINED MODEL COMPARISON
# Models: M1 (SIRS Benchmark) | M6 (Best Logistic Regression) | XGBoost
# Outputs: Calibration plots (windowed) + Decision Curve Analysis
# ============================================================

library(tidyverse)
library(tidymodels)
library(slider)
library(probably)
library(patchwork)
library(dcurves)    # install.packages("dcurves") if not already installed


# ============================================================
# SHARED VARIABLE LISTS
# ============================================================

excluded_variables <- c(
  "DBP", "TroponinI", "EtCO2", "PaCO2", "SaO2", "BaseExcess",
  "HCO3", "Hct", "AST", "Alkalinephos", "Bilirubin_direct",
  "Bilirubin_total", "PTT", "Fibrinogen", "Unit1", "Unit2"
)

sirs_variables <- c("patient_id", "HR", "Temp", "Resp", "WBC", "SepsisLabel", "ICULOS")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%
    ungroup()
}

add_indicator_columns <- function(df, outcome = "SepsisLabel") {
  no_indicator <- c("patient_id", "ICULOS", outcome)
  df %>%
    mutate(across(
      -any_of(no_indicator),
      ~ as.integer(is.na(.x)),
      .names = "was_na_{.col}"
    ))
}

add_obs_count_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      HR_obs_6h          = slide_int(HR,         ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Temp_obs_6h        = slide_int(Temp,       ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Resp_obs_6h        = slide_int(Resp,       ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      MAP_obs_6h         = slide_int(MAP,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      SBP_obs_6h         = slide_int(SBP,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      WBC_obs_6h         = slide_int(WBC,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      FiO2_obs_6h        = slide_int(FiO2,       ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Lactate_obs_6h     = slide_int(Lactate,    ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Platelets_obs_6h   = slide_int(Platelets,  ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Creatinine_obs_6h  = slide_int(Creatinine, ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      HR_obs_12h         = slide_int(HR,         ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Temp_obs_12h       = slide_int(Temp,       ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Resp_obs_12h       = slide_int(Resp,       ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      MAP_obs_12h        = slide_int(MAP,        ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      SBP_obs_12h        = slide_int(SBP,        ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      WBC_obs_12h        = slide_int(WBC,        ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      FiO2_obs_12h       = slide_int(FiO2,       ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Lactate_obs_12h    = slide_int(Lactate,    ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      total_obs_6h       = HR_obs_6h + Temp_obs_6h + Resp_obs_6h + MAP_obs_6h +
                           SBP_obs_6h + WBC_obs_6h + FiO2_obs_6h + Lactate_obs_6h +
                           Platelets_obs_6h + Creatinine_obs_6h
    ) %>%
    ungroup()
}

add_rolling_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      # --- 6h means ---
      HR_roll_mean_6         = slide_dbl(HR,         ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      Temp_roll_mean_6       = slide_dbl(Temp,       ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      Resp_roll_mean_6       = slide_dbl(Resp,       ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      MAP_roll_mean_6        = slide_dbl(MAP,        ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      SBP_roll_mean_6        = slide_dbl(SBP,        ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      WBC_roll_mean_6        = slide_dbl(WBC,        ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      FiO2_roll_mean_6       = slide_dbl(FiO2,       ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      Lactate_roll_mean_6    = slide_dbl(Lactate,    ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      Platelets_roll_mean_6  = slide_dbl(Platelets,  ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      Creatinine_roll_mean_6 = slide_dbl(Creatinine, ~mean(.x, na.rm = TRUE), .before = 5,  .complete = FALSE),
      # --- 12h means ---
      HR_roll_mean_12        = slide_dbl(HR,         ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Temp_roll_mean_12      = slide_dbl(Temp,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Resp_roll_mean_12      = slide_dbl(Resp,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      MAP_roll_mean_12       = slide_dbl(MAP,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      SBP_roll_mean_12       = slide_dbl(SBP,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      WBC_roll_mean_12       = slide_dbl(WBC,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      FiO2_roll_mean_12      = slide_dbl(FiO2,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Lactate_roll_mean_12   = slide_dbl(Lactate,    ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      # --- 6h standard deviations ---
      HR_roll_sd_6           = slide_dbl(HR,         ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_sd_6         = slide_dbl(Temp,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_sd_6         = slide_dbl(Resp,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_sd_6          = slide_dbl(MAP,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      SBP_roll_sd_6          = slide_dbl(SBP,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      WBC_roll_sd_6          = slide_dbl(WBC,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      FiO2_roll_sd_6         = slide_dbl(FiO2,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Lactate_roll_sd_6      = slide_dbl(Lactate,    ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Platelets_roll_sd_6    = slide_dbl(Platelets,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Creatinine_roll_sd_6   = slide_dbl(Creatinine, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE)
    ) %>%
    ungroup()
}


# ============================================================
# DATA PREPROCESSING
# ============================================================

# --- M1: SIRS benchmark (4 SIRS variables only) ---
train_m1 <- train %>%
  select(any_of(sirs_variables)) %>%
  ffill_dataset() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel))

test_m1 <- test %>%
  select(any_of(sirs_variables)) %>%
  ffill_dataset() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel))

# --- M6 + XGBoost: shared full preprocessing pipeline ---
train_full <- train %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns() %>%
  add_obs_count_features() %>%
  ffill_dataset() %>%
  add_rolling_features() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel),
         Gender      = as.factor(Gender))

test_full <- test %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns() %>%
  add_obs_count_features() %>%
  ffill_dataset() %>%
  add_rolling_features() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel),
         Gender      = as.factor(Gender))


# ============================================================
# RECIPES
# ============================================================

# M1: SIRS benchmark — logistic regression on SIRS_score only
recipe_m1 <- recipe(SepsisLabel ~ HR + Temp + Resp + WBC, data = train_m1) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) +
                           (Resp > 20) + (WBC > 12 | WBC < 4)) %>%
  step_select(SepsisLabel, SIRS_score)

# M6 + XGBoost share the same recipe (full feature set with SIRS score)
recipe_full <- recipe(SepsisLabel ~ ., data = train_full) %>%
  step_rm(patient_id) %>%
  step_dummy(Gender) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) +
                           (Resp > 20) + (WBC > 12 | WBC < 4)) %>%
  step_corr(threshold = 0.8)


# ============================================================
# MODEL ENGINES
# ============================================================

logistic_engine <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

xgb_engine <- boost_tree(
  trees      = 500,
  tree_depth = 7,
  learn_rate = 0.01209315,
  stop_iter  = 20
) %>%
  set_engine("xgboost", counts = FALSE, eval_metric = "auc", validation = 0.1) %>%
  set_mode("classification")


# ============================================================
# FIT MODELS & EXTRACT PREDICTIONS
# ============================================================

# --- M1: SIRS benchmark ---
message("[1/6] Prepping M1 recipe...")
prep_m1         <- recipe_m1 %>% prep(training = train_m1)
train_m1_baked  <- prep_m1   %>% bake(new_data = NULL)
test_m1_baked   <- prep_m1   %>% bake(new_data = test_m1)

message("[2/6] Fitting M1 (SIRS benchmark)...")
fit_m1   <- logistic_engine %>% fit(SepsisLabel ~ ., data = train_m1_baked)
pred_m1  <- predict(fit_m1, new_data = test_m1_baked, type = "prob")$.pred_1

# --- M6: Best logistic regression ---
# step_corr on 100+ features over the full training set is expensive — only run once.
# train_full_baked and test_full_baked are reused by XGBoost below.
message("[3/6] Prepping full recipe (step_corr — this is the slow step)...")
prep_full        <- recipe_full %>% prep(training = train_full)
train_full_baked <- prep_full   %>% bake(new_data = NULL)
test_full_baked  <- prep_full   %>% bake(new_data = test_full)

message("[4/6] Fitting M6 (logistic regression)...")
fit_m6   <- logistic_engine %>% fit(SepsisLabel ~ ., data = train_full_baked)
pred_m6  <- predict(fit_m6, new_data = test_full_baked, type = "prob")$.pred_1

# --- XGBoost ---
# IMPORTANT: use add_formula() + already-baked data (train_full_baked / test_full_baked)
# so that step_corr is NOT recomputed a second time inside the workflow.
message("[5/6] Fitting XGBoost...")
set.seed(123)
xgb_wf <- workflow() %>%
  add_formula(SepsisLabel ~ .) %>%
  add_model(xgb_engine)

fit_xgb  <- xgb_wf %>% fit(data = train_full_baked)
pred_xgb <- predict(fit_xgb, new_data = test_full_baked, type = "prob")$.pred_1

message("[6/6] All models fitted. Building plots...")


# ============================================================
# SHARED GROUND TRUTH
# ============================================================
# All models are evaluated on the same test observations;
# SepsisLabel is consistent across test_m1 and test_full.

# Factor with positive class ("1") first — used for calibration plots
truth_factor <- factor(as.numeric(as.character(test_m1$SepsisLabel)),
                       levels = c(1, 0))

# Numeric 0/1 — used for decision curve analysis
truth_num <- as.numeric(as.character(test_m1$SepsisLabel))


# ============================================================
# CALIBRATION PLOTS (windowed)
# ============================================================

make_cal_df <- function(pred_probs, truth) {
  tibble(.pred_1 = pred_probs, SepsisLabel = truth)
}

plot_cal_windowed <- function(df, title) {
  cal_plot_windowed(
    df,
    truth       = SepsisLabel,
    estimate    = .pred_1,
    event_level = "first",
    step_size   = 0.025,
    window_size = 0.03
  ) +
    ggtitle(title) +
    theme_bw(base_size = 10) +
    theme(
      plot.title       = element_text(face = "bold", size = 10),
      panel.grid.minor = element_blank()
    )
}

p_cal_m1  <- plot_cal_windowed(make_cal_df(pred_m1,  truth_factor), "M1: SIRS Benchmark")
p_cal_m6  <- plot_cal_windowed(make_cal_df(pred_m6,  truth_factor), "M6: Best Logistic Regression")
p_cal_xgb <- plot_cal_windowed(make_cal_df(pred_xgb, truth_factor), "XGBoost")

(p_cal_m1 | p_cal_m6 | p_cal_xgb) +
  plot_annotation(
    title = "Windowed Calibration Plots – Three-Model Comparison",
    theme = theme(plot.title = element_text(face = "bold", size = 13))
  )


# ============================================================
# DECISION CURVE ANALYSIS
# ============================================================

dca_df <- tibble(
  SepsisLabel    = truth_num,
  SIRS_Benchmark = pred_m1,
  Best_LogReg    = pred_m6,
  XGBoost        = pred_xgb
)

dca_result <- dca(
  SepsisLabel ~ SIRS_Benchmark + Best_LogReg + XGBoost,
  data       = dca_df,
  thresholds = seq(0, 0.60, by = 0.005)
)

plot(dca_result, smooth = TRUE) +
  labs(
    title = "Decision Curve Analysis – Three-Model Comparison",
    x     = "Threshold Probability",
    y     = "Net Benefit"
  ) +
  theme_bw() +
  theme(
    plot.title   = element_text(face = "bold", size = 13),
    legend.title = element_blank()
  )

# ============================================================
# SHAP FEATURE IMPORTANCE  (shapviz + kernelshap)
# install.packages(c("shapviz", "kernelshap"))  if needed
# ============================================================
library(shapviz)
library(kernelshap)

# Shared predict function for parsnip GLM fits (returns numeric prob vector)
pred_fn_glm <- function(object, newdata) {
  predict(object, new_data = newdata, type = "prob")$.pred_1
}

# ── XGBoost: native tree SHAP (fast, exact, no sampling needed) ──────────

xgb_core  <- extract_fit_engine(fit_xgb)
X_xgb     <- test_full_baked %>% select(-SepsisLabel) %>% as.matrix()
shap_xgb  <- shapviz(xgb_core, X_pred = X_xgb)

p_imp_xgb <- sv_importance(shap_xgb, show_numbers = TRUE) +
  ggtitle("XGBoost – SHAP Feature Importance") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10))

# ── M6 logistic: kernelshap (subsample test set to keep runtime manageable) ──
set.seed(42)
X_m6_explain <- test_full_baked  %>% select(-SepsisLabel) %>% slice_sample(n = 500)
X_m6_bg      <- train_full_baked %>% select(-SepsisLabel) %>% slice_sample(n = 100)

ks_m6    <- kernelshap(fit_m6, X = X_m6_explain, bg_X = X_m6_bg, pred_fun = pred_fn_glm)
shap_m6  <- shapviz(ks_m6)

p_imp_m6 <- sv_importance(shap_m6, show_numbers = TRUE) +
  ggtitle("M6: Best Logistic Regression – SHAP Feature Importance") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10))

# ── M1 SIRS benchmark: kernelshap (single feature – SIRS_score) ──────────

set.seed(42)
X_m1_explain <- test_m1_baked  %>% select(-SepsisLabel) %>% slice_sample(n = 500)
X_m1_bg      <- train_m1_baked %>% select(-SepsisLabel) %>% slice_sample(n = 100)

ks_m1    <- kernelshap(fit_m1, X = X_m1_explain, bg_X = X_m1_bg, pred_fun = pred_fn_glm)
shap_m1  <- shapviz(ks_m1)

p_imp_m1 <- sv_importance(shap_m1, show_numbers = TRUE) +
  ggtitle("M1: SIRS Benchmark – SHAP Feature Importance") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 10))

# ── Combined importance plot ──────────────────────────────────────────────
(p_imp_m1 | p_imp_m6 | p_imp_xgb) +
  plot_annotation(
    title = "SHAP Feature Importance – Three-Model Comparison",
    theme = theme(plot.title = element_text(face = "bold", size = 13))
  )