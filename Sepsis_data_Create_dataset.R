library(tidyverse)

#https://www.geeksforgeeks.org/r-language/read-all-files-in-directory-using-r/
#list.files finder alle filerne i mappen med path=x, full.names = true betyder at den returnere hele filstien på filen
training_set_a <- list.files(path="D:/physionet_training/training/training_setA", full.names = TRUE)
training_set_b <- list.files(path="D:/physionet_training/training/training_setB", full.names = TRUE)

#https://www.statology.org/combine-lists-in-r/
#c() combinere de to lister af paths(filer/patienter)
total_dataset <- c(training_set_a, training_set_b)

#https://rstudio.github.io/cheatsheets/html/data-import.html
#opretter en function der tager en variabel der hedder path som input
#read_delim() læser filen da det er en pipe-sperated, col_types=cols() lader den selv gætte datatype
read_patient_psv <- function(path) {
  read_delim(file = path, delim = "|",col_types = cols())
}


#map_dfr køre den samme funktion på alle elementerne i listen (alle 40336 filer)
#altså for hver fil køres funktionen read_patient_psv
#De samles så i en stor tabel med en ny kolonne kaldet "patient_id", alle rækker i fra samme fil får samme id
sepsis_data <- map_dfr(total_dataset, read_patient_psv, .id = "patient_id")

#sikrer at alle id "numrene" i kolonnen "patient_id" er integers
sepsis_data <- sepsis_data %>% mutate(patient_id = as.integer(patient_id))

#-----------------Outdated---------------------
#sepsis_data <- sepsis_data %>%
#  group_by(patient_id) %>%
#  mutate(hour = row_number()-1) %>%
#  ungroup()


#sepsis_data <- sepsis_data %>%
#  select(hour, everything())

#sepsis_data <- sepsis_data[, -which(names(sepsis_data) == "hour")]
#--------------------------------------

write_csv(sepsis_data, "sepsis_data.csv", na = "")


length(unique(sepsis_data$patient_id))
length(sepsis_data)

