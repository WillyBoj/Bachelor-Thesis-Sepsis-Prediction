library(tidymodels)

# recipe step_corr
sepsis_cor_rec <- recipe(SepsisLabel ~ ., data = train_data) %>%
  step_corr(all_numeric(), threshold = 0.8)

sepsis_cor_rec_prep <- prep(sepsis_cor_rec)
sepsis_cor_rec_prep

#chatGPT suggestion of correlation pairs (
cor_pairs <- train_data %>%
  select(where(is.numeric), -SepsisLabel) %>%   # kun numeriske predictors
  cor(use = "pairwise.complete.obs") %>%
  as.data.frame() %>%
  tibble::rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "correlation") %>%
  filter(var1 < var2) %>%                       # undgå dubletter
  arrange(desc(abs(correlation)))
#)
cor_pairs

#https://stackoverflow.com/questions/67247463/percentage-of-missing-data-in-multiple-variables-in-r
missing_summary <- train_data %>% summarize(across(c(everything()), ~mean(is.na(.x))*100))
