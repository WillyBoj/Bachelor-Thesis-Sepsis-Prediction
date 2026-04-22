library(tidyverse)
library(tidymodels)
library(slider)

# ----Exclusions & preprocessing ------------------

excluded_variables <- c(
  "DBP", "TroponinI", "EtCO2", "PaCO2", "SaO2", "BaseExcess",
  "HCO3", "Hct", "AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total",
  "PTT", "Fibrinogen", "Unit1", "Unit2"
)

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%
    ungroup()
}

add_indicator_columns <- function(df, outcome = "SepsisLabel") {
  No_indicator_variables <- c("patient_id", "ICULOS", outcome)
  df %>%
    mutate(across(
      -any_of(No_indicator_variables),
      ~ as.integer(is.na(.x)),
      .names = "was_na_{.col}"
    ))
}

add_obs_count_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      HR_obs_6h       = slide_int(HR,      ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Temp_obs_6h     = slide_int(Temp,    ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Resp_obs_6h     = slide_int(Resp,    ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      MAP_obs_6h      = slide_int(MAP,     ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      SBP_obs_6h      = slide_int(SBP,     ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      WBC_obs_6h      = slide_int(WBC,     ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      
      FiO2_obs_6h     = slide_int(FiO2,    ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Lactate_obs_6h  = slide_int(Lactate, ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      
      Platelets_obs_6h   = slide_int(Platelets,  ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Creatinine_obs_6h  = slide_int(Creatinine, ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      
      HR_obs_12h      = slide_int(HR,      ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Temp_obs_12h    = slide_int(Temp,    ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Resp_obs_12h    = slide_int(Resp,    ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      MAP_obs_12h     = slide_int(MAP,     ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      SBP_obs_12h     = slide_int(SBP,     ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      WBC_obs_12h     = slide_int(WBC,     ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      
      FiO2_obs_12h    = slide_int(FiO2,    ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Lactate_obs_12h = slide_int(Lactate, ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      
      total_obs_6h    = HR_obs_6h + Temp_obs_6h + Resp_obs_6h + MAP_obs_6h +
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
      HR_roll_mean_6          = slide_dbl(HR,         ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_mean_6        = slide_dbl(Temp,       ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_mean_6        = slide_dbl(Resp,       ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_mean_6         = slide_dbl(MAP,        ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      SBP_roll_mean_6         = slide_dbl(SBP,        ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      WBC_roll_mean_6         = slide_dbl(WBC,        ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      FiO2_roll_mean_6        = slide_dbl(FiO2,       ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Lactate_roll_mean_6     = slide_dbl(Lactate,    ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Platelets_roll_mean_6   = slide_dbl(Platelets,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Creatinine_roll_mean_6  = slide_dbl(Creatinine, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      
      # --- 12h means ---
      HR_roll_mean_12         = slide_dbl(HR,         ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Temp_roll_mean_12       = slide_dbl(Temp,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Resp_roll_mean_12       = slide_dbl(Resp,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      MAP_roll_mean_12        = slide_dbl(MAP,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      SBP_roll_mean_12        = slide_dbl(SBP,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      WBC_roll_mean_12        = slide_dbl(WBC,        ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      FiO2_roll_mean_12       = slide_dbl(FiO2,       ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Lactate_roll_mean_12    = slide_dbl(Lactate,    ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      
      # --- 6h standard deviations ---
      HR_roll_sd_6            = slide_dbl(HR,         ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_sd_6          = slide_dbl(Temp,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_sd_6          = slide_dbl(Resp,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_sd_6           = slide_dbl(MAP,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      SBP_roll_sd_6           = slide_dbl(SBP,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      WBC_roll_sd_6           = slide_dbl(WBC,        ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      FiO2_roll_sd_6          = slide_dbl(FiO2,       ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Lactate_roll_sd_6       = slide_dbl(Lactate,    ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Platelets_roll_sd_6     = slide_dbl(Platelets,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Creatinine_roll_sd_6    = slide_dbl(Creatinine, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE)
    ) %>%
    ungroup()
}
# ------Apply preprocessing -----------------------------------------------------

train_preprocess_ffill <- train %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  add_obs_count_features() %>% 
  ffill_dataset() %>%
  add_rolling_features() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel),
         Gender = as.factor(Gender))

test_preprocess_ffill <- test %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  add_obs_count_features() %>% 
  ffill_dataset() %>%
  add_rolling_features() %>%
  mutate(SepsisLabel = as.factor(SepsisLabel),
         Gender = as.factor(Gender))

#-----------------------------------------------------
# -------------------Recipe ----------------------
#-----------------------------------------------------
  
sepsis_recipe_xgb <- recipe(SepsisLabel ~ ., data = train_preprocess_ffill) %>%
  step_rm(patient_id) %>%
  step_dummy(Gender) %>%
  #step_impute_mean(all_numeric_predictors()) %>%
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) + (Resp > 20) + (WBC > 12 | WBC < 4)) %>%
  step_corr(threshold = 0.8)

# Tuning model  -----------------------------------------------------
xgb_tune_model <- boost_tree(
  trees       = 500,
  tree_depth  = tune(),
  learn_rate  = tune(),
  min_n       = tune(),
  sample_size = 0.8,
  mtry        = 0.8,
  stop_iter   = 20
) %>%
  set_engine("xgboost", counts = FALSE, eval_metric = "auc", validation = 0.1) %>%
  set_mode("classification")

# Setting up a tuning grid  -----------------------------------------------------
set.seed(123)
xgb_grid <- grid_latin_hypercube(
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(0.01, 0.2), trans = NULL),
  min_n(range = c(1, 10)),
  size = 20
)

# ----- setting up Workflow  -----------------------------------------------------
xgb_tune_wf <- workflow() %>%
  add_recipe(sepsis_recipe_xgb) %>%
  add_model(xgb_tune_model)

# setting up folds based on patient id in the training set (5 folds) ---------------------------
cv_folds <- group_vfold_cv(
  train_preprocess_ffill,
  group = patient_id,
  v     = 5
)

# Tuning the model  -----------------------------------------------------
xgb_tune_results <- xgb_tune_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid      = xgb_grid,
    metrics   = metric_set(roc_auc, brier_class),
    control   = control_grid(verbose = TRUE)
  )

# gets the 10 best results  -----------------------------------------------------
show_best(xgb_tune_results, metric = "roc_auc", n = 10)

# Selects the best parameters without over or underfitting  ----------------------
best_params <- select_best(xgb_tune_results, metric = "roc_auc")

# fitting  -----------------------------------------------------
xgb_final_wf <- xgb_tune_wf %>%
  finalize_workflow(best_params)

xgb_final_fit <- xgb_final_wf %>%
  fit(data = train_preprocess_ffill)

prob_preds <- predict(xgb_final_fit, new_data = test_preprocess_ffill, type = "prob")

sepsis_results <- test_preprocess_ffill %>%
  select(SepsisLabel) %>%
  bind_cols(prob_preds) %>%
  mutate(SepsisLabel = factor(SepsisLabel, levels = c("1", "0")))

#results -----------------------------------------------------
metric_set(roc_auc, brier_class)(sepsis_results, truth = SepsisLabel, .pred_1, event_level = "first")