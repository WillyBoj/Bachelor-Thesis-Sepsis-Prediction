library(tidyverse)
library(tidymodels)
library(ggcorrplot)
library(slider)
#exclusion of variables --------------------------------------------------------
excluded_variables <- c(
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

#------------------------------------------------------------------------
#Defining forward fill function-----------------------------------------

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%   # forward-fill
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

#------------------------------------------------------------------------
#defining add rolling features to dataset function -----------------------

add_rolling_features <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    mutate(
      # 6h means
      HR_roll_mean_6   = slide_dbl(HR,   ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_mean_6 = slide_dbl(Temp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_mean_6 = slide_dbl(Resp, ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_mean_6  = slide_dbl(MAP,  ~mean(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
    
      # change in variable over the past 3 hours
      HR_delta_3   = replace_na(HR   - lag(HR, 3),   0),
      Temp_delta_3 = replace_na(Temp - lag(Temp, 3), 0),
      Resp_delta_3 = replace_na(Resp - lag(Resp, 3), 0),
      MAP_delta_3  = replace_na(MAP  - lag(MAP, 3),  0),
      
      # 6h rolling sd window
      HR_roll_sd_6   = slide_dbl(HR,   ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      MAP_roll_sd_6  = slide_dbl(MAP,  ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Resp_roll_sd_6 = slide_dbl(Resp, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE),
      Temp_roll_sd_6 = slide_dbl(Temp, ~sd(.x, na.rm = TRUE), .before = 5, .complete = FALSE)
      
    ) %>%
    ungroup()
}

#------------------------------------------------------------------------

#Applying the function to preprocessing of the training and test dataset

train_preprocess_ffill <- train %>%                    
  select(-any_of(excluded_variables)) %>%                 #removing excluded variables
  add_indicator_columns(outcome = "SepsisLabel") %>%      #adding missing data indicators
  ffill_dataset() %>%                                     #Forward filling the missing data
  add_rolling_features() %>%                              #adding rolling features
  mutate(SepsisLabel = as.factor(SepsisLabel)) %>%        #making sure
  mutate(Gender = as.factor(Gender))


test_preprocess_ffill <- test %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  ffill_dataset() %>%
  add_rolling_features() %>% 
  mutate(SepsisLabel = as.factor(SepsisLabel))%>%        #making sure
  mutate(Gender = as.factor(Gender))

#train_preprocess_ffill %>% count(SepsisLabel)

#--------------------------------------------------------
#----------------recipe----------------------------------
#---------------------------------------------------------

sepsis_recipe_log <- recipe(SepsisLabel ~ ., data = train_preprocess_ffill) %>%
  step_rm(patient_id) %>%
  step_dummy(Gender) %>% 
  step_impute_mean(all_numeric_predictors()) %>%
  #step_mutate(shock_index = HR/SBP) %>%
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) + (Resp > 20) + (WBC > 12 | WBC < 4)) %>%  #creates new column with sirs scores based on the true arguments
  step_corr(threshold = 0.8)

#Model - mode and Engine---------------------------------

logistic_sepsis_model <- logistic_reg() %>%
  set_engine('glm') %>%
  set_mode('classification')
#-----------------------------------------------------------

#Prep and baking of data into Training and Test dataset ----
Sepsis_log_prep <- sepsis_recipe_log %>%
  prep(training = train_preprocess_ffill)

Sepsis_log_training_prep <- Sepsis_log_prep %>% 
  bake(new_data = NULL)

Sepsis_log_test_prep <- Sepsis_log_prep %>%
  bake(new_data = test_preprocess_ffill)

#-----------------------------------------------------------

# Fitting model -------------------------------------------
Logistic_sepsis_fit <- logistic_sepsis_model %>%
  fit(SepsisLabel~., data = Sepsis_log_training_prep)

#-----------------------------------------------------------

#preparing preditions --------------------------------------------------------
Class_preds_sepsis <- predict(Logistic_sepsis_fit, new_data = Sepsis_log_test_prep, type = "class")

prob_preds_sepsis <- predict(Logistic_sepsis_fit, new_data = Sepsis_log_test_prep, type = "prob")
#-----------------------------------------------------------

#Results of the model --------------------------------------
sepsis_results <- test_preprocess_ffill %>%
  select(SepsisLabel) %>%
  bind_cols(Class_preds_sepsis, prob_preds_sepsis)

sepsis_results
#-----------------------------------------------------------

# test_preprocess_ffill <- test_preprocess_ffill %>%
#  mutate(Sepsis_Preds = prob_preds_sepsis$.pred_1*100)


#Evalutation --------------------------------------------------
# library(DescTools)
# BrierScore(sepsis_results$SepsisLabel == "1", sepsis_results$.pred_1)

sepsis_results$SepsisLabel <- factor(sepsis_results$SepsisLabel, levels = c("1", "0")) #yardstick fix https://github.com/tidymodels/yardstick/issues/515

prediction_eval <- metric_set(roc_auc, brier_class)

prediction_eval(sepsis_results, truth = SepsisLabel, .pred_1, event_level = "first")
#-----------------------------------------------------------

