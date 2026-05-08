library(tidyverse)

# ── 1. Variable order and labels ─────────────────────────────────────────────
plot_vars <- c(
  "HR", "MAP", "SBP", "O2Sat",
  "Resp", "FiO2", "BUN", "Creatinine",
  "Magnesium", "Calcium", "Chloride", "Phosphate",
  "Potassium", "Lactate", "pH", "Hgb",
  "WBC", "Platelets", "Glucose", "Temp",
  "Age", "Gender", "HospAdmTime", "ICULOS"
)

var_labels <- c(
  HR          = "Heart Rate (bpm)",
  MAP         = "Mean Arterial Pressure (mmHg)",
  SBP         = "Systolic BP (mmHg)",
  O2Sat       = "Oxygen Saturation (%)",
  Resp        = "Respiration Rate (breaths/min)",
  FiO2        = "Fraction of Inspired O2 (%)",
  BUN         = "Blood Urea Nitrogen (mg/dL)",
  Creatinine  = "Creatinine (mg/dL)",
  Magnesium   = "Magnesium (mmol/dL)",
  Calcium     = "Calcium (mg/dL)",
  Chloride    = "Chloride (mmol/L)",
  Phosphate   = "Phosphate (mg/dL)",
  Potassium   = "Potassium (mmol/L)",
  Lactate     = "Lactic Acid (mg/dL)",
  pH          = "pH",
  Hgb         = "Hemoglobin (g/dL)",
  WBC         = "Leukocyte Count (count/L)",
  Platelets   = "Platelet Count (count/mL)",
  Glucose     = "Serum Glucose (mg/dL)",
  Temp        = "Temperature (°C)",
  Age         = "Age (years)",
  Gender      = "Gender (0=F, 1=M)",
  HospAdmTime = "Hospital to ICU Time (hours)",
  ICULOS      = "ICU Length of Stay (hours)"
)

# ── 2. X-axis limits ──────────────────────────────────────────────────────────
xlims <- tibble::tribble(
  ~variable,    ~xmin,  ~xmax,
  "HR",          0,      200,
  "MAP",         0,      150,
  "SBP",         50,     250,
  "O2Sat",       80,     100,
  "Resp",        0,      50,
  "FiO2",        0,      1,
  "BUN",         0,      100,
  "Creatinine",  0,      10,
  "Magnesium",   0,      5,
  "Calcium",     5,      15,
  "Chloride",    80,     130,
  "Phosphate",   0,      10,
  "Potassium",   2,      8,
  "Lactate",     0,      10,
  "pH",          6.75,   7.75,
  "Hgb",         0,      20,
  "WBC",         0,      40,
  "Platelets",   0,      600,
  "Glucose",     0,      400,
  "Temp",        32.5,   42.5,
  "Age",         0,      100,
  "Gender",     -0.5,    1.5,
  "HospAdmTime",-100,    50,
  "ICULOS",      0,      150
)

# ── 3. Pivot to long and clip values to x limits ──────────────────────────────
sepsis_long <- sepsis_data %>%
  select(all_of(c(plot_vars, "SepsisLabel"))) %>%
  pivot_longer(
    cols      = all_of(plot_vars),
    names_to  = "variable",
    values_to = "value"
  ) %>%
  left_join(xlims, by = "variable") %>%
  filter(value >= xmin, value <= xmax) %>%   # <-- clips instead of scale tricks
  select(-xmin, -xmax) %>%
  mutate(variable = factor(variable, levels = plot_vars))

# ── 4. Plot ───────────────────────────────────────────────────────────────────
sepsis_long %>%
  ggplot(aes(
    x      = value,
    colour = factor(SepsisLabel),
    fill   = factor(SepsisLabel)
  )) +
  geom_density(alpha = 0.4) +
  facet_wrap(
    ~ variable,
    ncol     = 4,
    scales   = "free",
    labeller = as_labeller(var_labels)
  ) +
  scale_colour_manual(
    values = c("0" = "#F8766D", "1" = "#00BFC4"),
    labels = c("Not sepsis", "Sepsis")
  ) +
  scale_fill_manual(
    values = c("0" = "#F8766D", "1" = "#00BFC4"),
    labels = c("Not sepsis", "Sepsis")
  ) +
  labs(
    title  = "Density distributions per variable stratified by Sepsis status",
    x      = NULL,
    y      = "Density",
    colour = "Sepsis status",
    fill   = "Sepsis status"
  ) +
  theme_light() +
  theme(
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    strip.text       = element_text(size = 8, colour = "black"),
    strip.background = element_rect(fill = "grey92", colour = NA),
    axis.text        = element_text(size = 6),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(size = 11)
  )
