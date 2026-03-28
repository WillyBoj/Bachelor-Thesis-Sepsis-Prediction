library(tidyverse)
library(tidymodels)

#exclusion of variables --------------------------------------------------------
excluded_variables <- c(
  "SBP",
  "DBP",
  "Hct",
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
  "Phosphate",
  "Magnesium",
  "Calcium",
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

#Applying the functions to preprocessing of the training and test dataset

train_preprocess_ffill <- train %>%
  select(-any_of(excluded_variables)) %>%
  ffill_dataset() %>% 
  mutate(SepsisLabel = as.factor(SepsisLabel))

test_preprocess_ffill <- test %>%
  select(-any_of(excluded_variables)) %>%
  ffill_dataset() %>% 
  mutate(SepsisLabel = as.factor(SepsisLabel))


#--------------------------------------------------------
#----------------recipe----------------------------------
#---------------------------------------------------------

SIRS_recipe <- recipe(SepsisLabel ~ HR + Temp + Resp + WBC, data = train_preprocess_ffill) %>%            #selecting the 4 variables used in the SIRS model
  step_impute_mean(all_numeric_predictors()) %>%                                                          #uses the mean from training dataset to fill the rest of NA
  step_mutate(SIRS_score = (HR > 90) + (Temp > 38 | Temp < 36) + (Resp > 20) + (WBC > 12 | WBC < 4)) %>%  #creates new column with sirs scores based on the true arguments
  step_select(SepsisLabel, SIRS_score)                                                                    #selects SepsisLabel and SIRS_score as variables

#Model - mode and Engine---------------------------------

logistic_sirs_model <- logistic_reg() %>%
  set_engine('glm') %>%
  set_mode('classification')
#-----------------------------------------------------------

#Prep and baking of data into Training and Test dataset ----

Sirs_benchmark_prep <- SIRS_recipe %>%
  prep(training = train_preprocess_ffill)

Sirs_benchmark_training_prep <- Sirs_benchmark_prep %>%
  bake(new_data = NULL)

Sirs_benchmark_test_prep <- Sirs_benchmark_prep %>%
  bake(new_data = test_preprocess_ffill)
#-----------------------------------------------------------

# Fitting model -------------------------------------------
Logistic_sirs_fit <- logistic_sirs_model %>%
  fit(SepsisLabel~SIRS_score, data = Sirs_benchmark_training_prep)
#-----------------------------------------------------------

#preparing preditions --------------------------------------------------------
Class_preds_sirs <- predict(Logistic_sirs_fit, new_data = Sirs_benchmark_test_prep, type = "class")

prob_preds_sirs <- predict(Logistic_sirs_fit, new_data = Sirs_benchmark_test_prep, type = "prob")
#-----------------------------------------------------------

#Results of the model --------------------------------------
sirs_results <- test_preprocess_ffill %>%
  select(SepsisLabel) %>%
  bind_cols(Class_preds_sirs, prob_preds_sirs)

test_preprocess_ffill <- test_preprocess_ffill %>%
  mutate(SIRS_Preds = prob_preds_sirs$.pred_1*100)

sirs_results
#-----------------------------------------------------------

#Evalutation --------------------------------------------------
sirs_results$SepsisLabel <- factor(sirs_results$SepsisLabel, levels = c("1", "0")) #yardstick fix https://github.com/tidymodels/yardstick/issues/515

prediction_eval <- metric_set(roc_auc, brier_class)

prediction_eval(sirs_results, truth = SepsisLabel,.pred_1, event_level = "first")

#-----------------------------------------------------------

