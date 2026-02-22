library(tidyverse)
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
missing_summary <- percent_missing(df_imp)
#--------------------------------------------------