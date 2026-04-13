library(tidyverse)
library(tidymodels)
library(slider)

#exclusion of variables --------------------------------------------------------

excluded_variables <- c(
  "DBP", "TroponinI", "EtCO2", "PaCO2", "SaO2", "BaseExcess",
  "HCO3", "Hct", "AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total",
  "PTT", "Fibrinogen", "Unit1", "Unit2"
)
#------------------------------------------------------------------------
#Defining forward fill function-----------------------------------------

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%
    ungroup()
}

#------------------------------------------------------------------------
#defining missing indicator creation function ---------------------------

add_indicator_columns <- function(df, outcome = "SepsisLabel") {
  No_indicator_variables <- c("patient_id", "ICULOS", outcome)
  df %>%
    mutate(across(
      -any_of(No_indicator_variables),
      ~ as.integer(is.na(.x)),
      .names = "was_na_{.col}"
    ))
}

# Counts how many times each variable was measured in the last 6 and 12 hours.
# Few measurements may indicate a less monitored or more stable patient.
#------------------------------------------------------------------------

add_obs_count_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      HR_obs_6h          = slide_int(HR,         ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Temp_obs_6h        = slide_int(Temp,       ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Resp_obs_6h        = slide_int(Resp,       ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      MAP_obs_6h         = slide_int(MAP,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      FiO2_obs_6h         = slide_int(FiO2,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      Lactate_obs_6h         = slide_int(Lactate,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      SBP_obs_6h         = slide_int(SBP,        ~sum(!is.na(.x)), .before = 5,  .complete = FALSE),
      
      HR_obs_12h         = slide_int(HR,         ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      Temp_obs_12h       = slide_int(Temp,       ~sum(!is.na(.x)), .before = 11, .complete = FALSE),
      FiO2_obs_12h         = slide_int(FiO2,        ~sum(!is.na(.x)), .before = 11,  .complete = FALSE),
      Lactate_obs_12h         = slide_int(Lactate,        ~sum(!is.na(.x)), .before = 11,  .complete = FALSE),
      
      total_obs_6h       = HR_obs_6h + Temp_obs_6h + Resp_obs_6h + MAP_obs_6h + FiO2_obs_6h + Lactate_obs_6h + SBP_obs_6h ) %>%
    ungroup()
}

#------------------------------------------------------------------------
#defining add rolling features to dataset function -----------------------

add_rolling_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      HR_roll_mean_6   = slide_dbl(HR,   ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_mean_6 = slide_dbl(Temp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_mean_6 = slide_dbl(Resp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_mean_6  = slide_dbl(MAP,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Lactate_roll_mean_6  = slide_dbl(Lactate,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      FiO2_roll_mean_6  = slide_dbl(FiO2,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Platelets_roll_mean_6  = slide_dbl(Platelets,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Creatinine_roll_mean_6  = slide_dbl(Creatinine,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      
      
      HR_roll_mean_12   = slide_dbl(HR,   ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Temp_roll_mean_12 = slide_dbl(Temp, ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Resp_roll_mean_12 = slide_dbl(Resp, ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      MAP_roll_mean_12  = slide_dbl(MAP,  ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      Lactate_roll_mean_12  = slide_dbl(Lactate,  ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      FiO2_roll_mean_12  = slide_dbl(FiO2,  ~mean(.x, na.rm = TRUE), .before = 11, .complete = FALSE),
      
      HR_roll_sd_6   = slide_dbl(HR,   ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_sd_6  = slide_dbl(MAP,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_sd_6 = slide_dbl(Resp, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_sd_6 = slide_dbl(Temp, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Lactate_roll_sd_6   = slide_dbl(Lactate,   ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      FiO2_roll_sd_6   = slide_dbl(FiO2,   ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Platelets_roll_sd_6  = slide_dbl(Platelets,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Creatinine_roll_sd_6  = slide_dbl(Creatinine,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE)
      
    ) %>%
    ungroup()
}

#------------------------------------------------------------------------
#Applying the function to preprocessing of the training and test dataset

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

#--------------------------------------------------------
#----------------recipe----------------------------------
#---------------------------------------------------------

sepsis_recipe_xgb <- recipe(SepsisLabel ~ ., data = train_preprocess_ffill) %>%
  step_rm(patient_id) %>%
  step_dummy(Gender) %>%
  #step_impute_mean(all_numeric_predictors()) %>%
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) + (Resp > 20) + (WBC > 12 | WBC < 4)) %>%
  step_corr(threshold = 0.8)

#Model - Hyperparameters, mode and Engine---------------------------------

xgb_sepsis_model <- boost_tree(
  trees          = 500,
  tree_depth     = 6,
  learn_rate     = 0.05,
  min_n          = 5,
  sample_size    = 0.8,
  mtry           = 0.8,
  stop_iter      = 20
) %>%
  set_engine(
    "xgboost",
    counts           = FALSE,
    eval_metric      = "auc",
    validation       = 0.1
  ) %>%
  set_mode("classification")

#---- using Workflow to prep bake and add the model-----------------------------------

xgb_wf <- workflow() %>%
  add_recipe(sepsis_recipe_xgb) %>%
  add_model(xgb_sepsis_model)

# Fitting model -------------------------------------------

xgb_final_fit <- xgb_wf %>%
  fit(data = train_preprocess_ffill)

#-----------------------------------------------------------
#preparing preditions --------------------------------------------------------

class_preds <- predict(xgb_final_fit, new_data = test_preprocess_ffill, type = "class")
prob_preds  <- predict(xgb_final_fit, new_data = test_preprocess_ffill, type = "prob")

#-----------------------------------------------------------
#Results of the model --------------------------------------

sepsis_results <- test_preprocess_ffill %>%
  select(SepsisLabel) %>%
  bind_cols(class_preds, prob_preds) %>%
  mutate(SepsisLabel = factor(SepsisLabel, levels = c("1", "0")))

#-----------------------------------------------------------
#Evalutation --------------------------------------------------

prediction_eval <- metric_set(roc_auc, brier_class)

prediction_eval(sepsis_results, truth = SepsisLabel, .pred_1, event_level = "first")