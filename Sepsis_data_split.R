library(tidymodels)
tidymodels_prefer()

set.seed(501)

# Lav patient-level tabel til split (1 række pr patient)
# For hver patient, giv mig én række, og sæt SepsisLabel til den højeste værdi patienten
#fjern gruppering med "drop"
#
patients <- sepsis_data %>%
  group_by(patient_id) %>%
  summarise(SepsisLabel = max(SepsisLabel), .groups = "drop")

# Split på patient_id, 20% test 80% træning og strata på baggrund af sepsislavel
patient_split <- initial_split(patients, prop = 0.8, strata = SepsisLabel)

# kæd til datasæt
train_patients <- training(patient_split)
test_patients  <- testing(patient_split)

# koble id tilbage sammen med data med semi_join
# chatgpt suggestion()
train_data <- sepsis_data %>%
  semi_join(train_patients, by = "patient_id")

test_data <- sepsis_data %>%
  semi_join(test_patients, by = "patient_id")
#)