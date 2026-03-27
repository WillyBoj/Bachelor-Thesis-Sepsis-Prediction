library(tidyverse)
library(tidymodels)
library(ggcorrplot)
library(slider)
library(pROC)
#----------------recipe----------------------------------
#---------------------------------------------------------
excluded_variables <- c(
  "SBP",
  "DBP",
  "TroponinI",
  "EtCO2",
  "FiO2",
  "PaCO2",
  "SaO2",
  "BaseExcess",
  "HCO3",
  "pH",
  "Lactate",
  "Chloride",
  "Calcium",
  "Magnesium",
  "Hct",
  "Phosphate",
  "AST",
  "Alkalinephos",
  "Bilirubin_direct",
  "Bilirubin_total",
  "PTT",
  "Fibrinogen",
  "Unit1",
  "Unit2"
)

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%   # forward-fill
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



add_rolling_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      HR_roll_mean_6   = slide_dbl(HR,   ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_mean_6 = slide_dbl(Temp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_mean_6 = slide_dbl(Resp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      
      HR_roll_max_6    = slide_dbl(HR,   ~ifelse(all(is.na(.x)), NA_real_, max(.x, na.rm = TRUE)), .before = 5, .complete = FALSE),
      Temp_roll_max_6  = slide_dbl(Temp, ~ifelse(all(is.na(.x)), NA_real_, max(.x, na.rm = TRUE)), .before = 5, .complete = FALSE),
      
      HR_delta_3       = replace_na(HR   - lag(HR, 3),   0),
      Temp_delta_3     = replace_na(Temp - lag(Temp, 3), 0),
      Resp_delta_3     = replace_na(Resp - lag(Resp, 3), 0)
    ) %>%
    ungroup()
}


train_preprocess_ffill <- train %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  ffill_dataset() %>%
  add_rolling_features() %>% 
  mutate(SepsisLabel = as.factor(SepsisLabel))


test_preprocess_ffill <- test %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  ffill_dataset() %>%
  add_rolling_features() %>% 
  mutate(SepsisLabel = as.factor(SepsisLabel))


train_preprocess_ffill %>% count(SepsisLabel)


sepsis_recipe_log <- recipe(SepsisLabel ~ ., data = train_preprocess_ffill) %>%
  step_rm(patient_id) %>%
  step_impute_mean(all_numeric_predictors())

logistic_sepsis_model <- logistic_reg() %>%
  set_engine('glm') %>%
  set_mode('classification')

Sepsis_log_prep <- sepsis_recipe_log %>%
  prep(training = train_preprocess_ffill)

Sepsis_log_training_prep <- Sepsis_log_prep %>%   # ← Sepsis_log_prep, ikke sepsis_recipe_log
  bake(new_data = NULL)

Sepsis_log_test_prep <- Sepsis_log_prep %>%
  bake(new_data = test_preprocess_ffill)

Logistic_sepsis_fit <- logistic_sepsis_model %>%
  fit(SepsisLabel~., data = Sepsis_log_training_prep)

Class_preds_sepsis <- predict(Logistic_sepsis_fit, new_data = Sepsis_log_test_prep, type = "class")

prob_preds_sepsis <- predict(Logistic_sepsis_fit, new_data = Sepsis_log_test_prep, type = "prob")

sepsis_results <- test_preprocess_ffill %>%
  select(SepsisLabel) %>%
  bind_cols(Class_preds_sepsis, prob_preds_sepsis)

sepsis_results

test_preprocess_ffill <- test_preprocess_ffill %>%
  mutate(Sepsis_Preds = prob_preds_sepsis$.pred_1*100)

####CLAUDE####

# Get the ROC curve with all possible thresholds
roc_data <- roc_curve(sepsis_results, 
                      truth = SepsisLabel, 
                      .pred_1, 
                      event_level = "second")

# Find optimal threshold using Youden's J (maximizes sens + spec - 1)
roc_data <- roc_data %>%
  mutate(j_index = sensitivity + specificity - 1)

best_threshold <- roc_data %>%
  slice_max(j_index, n = 1)

best_threshold

sepsis_results <- sepsis_results %>%
  mutate(.pred_class_optimal = factor(
    ifelse(.pred_1 > best_threshold$.threshold, 1, 0),
    levels = levels(SepsisLabel)
  ))
##########

prediction_eval <- metric_set(accuracy, sens, spec, roc_auc, brier_class)

prediction_eval(sepsis_results, truth = SepsisLabel, estimate=.pred_class_optimal,.pred_1, event_level = "second")

conf_mat(sepsis_results, truth = SepsisLabel, estimate=.pred_class_optimal)
