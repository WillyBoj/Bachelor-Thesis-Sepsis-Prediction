library(readr)
library(tidymodels)

tidymodels_prefer()

sepsis_data <- read_csv("sepsis_data.csv")

#chapter about spending our data in TMWR
set.seed(501)

#tag sepsisdatasættet og find alle unikke 
patients <- sepsis_data %>%
  distinct(patient_id)

patient_split <- initial_split(
  patients,
  prop = 0.8
)

train_patients <- training(patient_split)
test_patients  <- testing(patient_split)

#ChatGPT suggestion(

train_data <- sepsis_data %>%
  semi_join(train_patients, by = "patient_id")

test_data <- sepsis_data %>%
  semi_join(test_patients, by = "patient_id")

#)