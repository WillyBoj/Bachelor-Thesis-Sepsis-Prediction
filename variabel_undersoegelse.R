library(tidyverse)
library(tidymodels)
library(ggcorrplot)
library(mde)
#--------- test data loading --------------------
df <- train
#--------------------------------------------------


#--------- Missing percentage --------------------
missing_summary <- df %>% summarize(across(c(everything()), ~mean(is.na(.x))*100))
#--------------------------------------------------

#--------- linear correlations --------------------
df_num <- df %>%
  select(is.numeric) %>%
  select(-patient_id, -SepsisLabel)

# correlation matrix (spearman is robust for ICU data)
cormat <- cor(df_num, use = "pairwise.complete.obs", method = "spearman")
cormat
# plot
ggcorrplot(cormat,
           hc.order = TRUE, type = "full",
           outline.color = "grey",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726")
)

#--------------------------------------------------

#--------% of patients with at least one measurement------------------------------------------

#made with ChatGPT(
patient_coverage <- function(df, id_col = "patient_id") {
  stopifnot(id_col %in% names(df))
df %>%
  select(all_of(id_col), everything()) %>%
  pivot_longer(
    cols = -all_of(id_col),
    names_to = "variable",
    values_to = "value"
  ) %>%
  group_by(variable) %>%
  summarise(
    patients_with_any_value = n_distinct(.data[[id_col]][!is.na(value)]),
    total_patients = n_distinct(.data[[id_col]]),
    pct_patients_with_any_value = (patients_with_any_value / total_patients)*100,
    .groups = "drop"
  ) %>%
  arrange(desc(pct_patients_with_any_value), desc(patients_with_any_value))
}

coverage_tbl <- patient_coverage(df, id_col = "patient_id")
#)
#--------------------------------------------------


#------------------drop columns-------------------------
df <- train
iteration1 <- df %>% select(-c(
  SBP,
  DBP,
  TroponinI,
  EtCO2,
  FiO2,
  PaCO2,
  SaO2,
  BaseExcess,
  HCO3,
  pH,
  Lactate,
  Chloride,
  Phosphate,
  Magnesium,
  Calcium,
  AST,
  Alkalinephos,
  Bilirubin_direct,
  Bilirubin_total,
  PTT,
  Fibrinogen,
  Unit1,
  Unit2
))
#--------------------------------------------------

#---------------Forward fill-----------------------------------
iteration1_ffill <- iteration1 %>%
  arrange(patient_id, ICULOS) %>%
  group_by(patient_id) %>%
  fill(everything(), .direction = "down") %>%   # forward-fill
  ungroup()
missing_summary <- percent_missing(iteration1_ffill)
#--------------------------------------------------

iteration1_mean_impute <- iteration1_ffill %>%
  mutate(across(
    where(is.numeric),
    ~ if_else(is.na(.x), mean(.x, na.rm = TRUE), .x)
  ))
missing_summary <- percent_missing(iteration1_mean_impute)



#---------------------------------------------------------
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

ffill_dataset <- function(df) {
  df %>%
    arrange(patient_id, ICULOS) %>%
    group_by(patient_id) %>%
    fill(-patient_id, -ICULOS, .direction = "down") %>%   # forward-fill
    ungroup()
}

add_indicator_columns <- function(df, outcome = "SepsisLabel") {
  protected <- c("patient_id", "ICULOS", outcome)
  
  df %>%
    mutate(across(
      -any_of(protected),
      ~ as.integer(is.na(.x)),
      .names = "was_na_{.col}"
    ))
}

train_preprocess_ffill <- train %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  ffill_dataset()

test_preprocess_ffill <- test %>%
  select(-any_of(excluded_variables)) %>%
  add_indicator_columns(outcome = "SepsisLabel") %>%
  ffill_dataset()

sepsis_recipe <- recipe(SepsisLabel~.,data = train_preprocess_ffill) %>%
  step_meanimpute(all_numeric_predictors()) %>%
  #step_corr(all_numeric(), threshold = 0.9) %>% Comment out because it will remove hct hgb, we are considering compositevariable
  #step_log() Datacamp course does this, we want to explore different transformations
  step_normalize(all_numeric_predictors())
  
  

  