# This script generates chord diagrams illustrating the relationships between neurons 
# and behavioral measures in the ACC or CLA brain regions. It:
#  1. Loads intertrial and asteroid appearance neuron-behavior data from Excel files.
#  2. Normalizes and categorizes behaviors into functional groups.
#  3. Defines plotting order and colors for behaviors.
#  4. Creates chord diagrams for intertrial, asteroid appearance, and shared neurons.
#  5. Saves the diagrams as SVG files for visualization of neuron-behavior interactions.
#
# This code generates:
#  Figures
#    4j, k
#    5g, h
#  Extended Data Figures
#    5g
#    6i
# Date:
# RStudio

# Load libraries
library(readxl)
library(dplyr)
library(circlize)

# Settings
brain_region <- 'ACC'  # Options: 'ACC' or 'CLA'
base_path <- "path/Results_MI/"

# Load intertrial and asteroid data
df_intertrial <- read_excel(paste0(base_path, "neuron_id_intertrial_MI_score_", brain_region, ".xlsx")) %>%
  rename(Behavior = behavior_name, NeuronID = file_name)
df_asteroid <- read_excel(paste0(base_path, "neuron_id_asteroid_MI_score_", brain_region, ".xlsx")) %>%
  rename(Behavior = behavior_name, NeuronID = file_name)

df_intertrial$Event <- "intertrial"
df_asteroid$Event <- "asteroid"

# Normalize behavior names by removing A_/B_ prefixes
normalize_behavior <- function(df) {
  df %>% mutate(Behavior = gsub("^[AB]_", "", Behavior))
}
df_intertrial <- normalize_behavior(df_intertrial)
df_asteroid <- normalize_behavior(df_asteroid)

# Assign behavior categories
assign_behavior_category <- function(df) {
  df %>%
    mutate(BehaviorCategory = case_when(
      grepl("absolute_prediction_error", Behavior) ~ "Prediction Error",
      grepl("safety_variance", Behavior) ~ "Uncertainty",
      grepl("outcome", Behavior) ~ "Safety Belt",
      grepl("y-Position", Behavior) ~ "y-Position",
      grepl("Outcome", Behavior) ~ "Outcome",
      TRUE ~ "Other"
    ))
}

# Define behavior order
all_behaviors <- unique(c(df_intertrial$Behavior, df_asteroid$Behavior))
fixed_behavior_order <- c(
  "absolute_prediction_error",
  "safety_variance",
  "outcome",
  "y-Position",
  "Outcome"
)
fixed_behavior_order <- fixed_behavior_order[fixed_behavior_order %in% all_behaviors]

# Chord plot function
plot_chord <- function(df, title_text, output_file, fixed_behavior_order = NULL) {
  df <- assign_behavior_category(df)
  
  df_summary <- df %>%
    group_by(NeuronID, Behavior, BehaviorCategory) %>%
    summarise(value = n(), .groups = "drop") %>%
    rename(from = NeuronID, to = Behavior)
  
  neuron_order <- unique(df_summary$from)
  
  if (is.null(fixed_behavior_order)) {
    behavior_order <- unique(df_summary$to)
  } else {
    behavior_order <- fixed_behavior_order[fixed_behavior_order %in% df_summary$to]
  }
  
  full_order <- c(neuron_order, behavior_order)
  
  behavior_colors <- df_summary %>%
    distinct(to, BehaviorCategory) %>%
    mutate(color = case_when(
      BehaviorCategory == "Prediction Error" ~ "#AF9BCA",
      BehaviorCategory == "Uncertainty"      ~ "#B1A0A1",
      BehaviorCategory == "Safety Belt"      ~ "#F15A29",
      BehaviorCategory == "y-Position"       ~ "#d62728",
      BehaviorCategory == "Outcome"          ~ "#687798",
      TRUE ~ "#999999"
    )) %>%
    filter(to %in% behavior_order) %>%
    slice(match(behavior_order, to))
  
  color_mapping <- c(
    setNames(rep("#000000", length(neuron_order)), neuron_order),
    setNames(behavior_colors$color, behavior_colors$to)
  )
  
  link_colors <- df_summary %>%
    left_join(behavior_colors, by = c("to" = "to")) %>%
    mutate(link_id = paste(from, to, sep = " - ")) %>%
    select(link_id, color)
  
  link_col_vector <- setNames(link_colors$color, link_colors$link_id)
  
  df_summary <- df_summary %>%
    mutate(value_scaled = ifelse(to %in% behavior_order, value * 3, value))  # Emphasize behavior sectors
  
  circos.clear()
  circos.par(start.degree = 90, gap.degree = 4, track.margin = c(0.01, 0.01), cell.padding = c(0, 0, 0, 0))
  
  svg(output_file, width = 10, height = 10)
  chordDiagram(df_summary[, c("from", "to", "value_scaled")],
               grid.col = color_mapping,
               order = full_order,
               col = link_col_vector,
               annotationTrack = "grid",
               transparency = 0.5,
               link.lwd = 0.3,
               link.border = NA,
               directional = 0,
               link.sort = TRUE,
               link.decreasing = FALSE,
               preAllocateTracks = list(track.height = 0.05))
  
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    sector.name = get.cell.meta.data("sector.index")
    xlim = get.cell.meta.data("xlim")
    ylim = get.cell.meta.data("ylim")
    circos.text(mean(xlim), ylim[1] + .2, sector.name, facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5), cex = 0.6)
  }, bg.border = NA)
  
  title(title_text)
  dev.off()
}

# 1. Plot chord for intertrial
plot_chord(
  df = df_intertrial,
  title_text = "Chord Diagram - Intertrial",
  output_file = paste0(base_path, "neuron_chord_intertrial_", brain_region, ".svg"),
  fixed_behavior_order = fixed_behavior_order
)

# 2. Plot chord for asteroid
plot_chord(
  df = df_asteroid,
  title_text = "Chord Diagram - Asteroid",
  output_file = paste0(base_path, "neuron_chord_asteroid_", brain_region, ".svg"),
  fixed_behavior_order = fixed_behavior_order
)

# 3. Plot shared neurons (intertrial vs asteroid)
shared_neurons <- intersect(df_intertrial$NeuronID, df_asteroid$NeuronID)

df_shared <- bind_rows(df_intertrial, df_asteroid) %>%
  filter(NeuronID %in% shared_neurons) %>%
  mutate(
    Behavior = paste(Behavior, Event, sep = "_"),
    BehaviorCategory = case_when(
      grepl("absolute_prediction_error", Behavior) ~ "Prediction Error",
      grepl("safety_variance", Behavior) ~ "Uncertainty",
      grepl("outcome", Behavior) ~ "Safety Belt",
      grepl("y-Position", Behavior) ~ "y-Position",
      grepl("Outcome", Behavior) ~ "Outcome",
      TRUE ~ "Other"
    )
  )

behavior_categories <- c("Prediction Error", "Uncertainty", "Safety Belt", "y-Position", "Outcome")
event_order <- c("intertrial", "asteroid")

fixed_behavior_order_shared <- unlist(lapply(behavior_categories, function(cat) {
  behaviors <- unique(df_shared$Behavior[df_shared$BehaviorCategory == cat])
  ordered <- unlist(lapply(event_order, function(ev) {
    grep(paste0("_", ev, "$"), behaviors, value = TRUE)
  }))
  ordered
}))

if (nrow(df_shared) > 0) {
  plot_chord(
    df = df_shared,
    title_text = "Chord Diagram - Shared Neurons Intertrial vs Asteroid",
    output_file = paste0(base_path, "neuron_chord_intertrial_asteroid_", brain_region, ".svg"),
    fixed_behavior_order = fixed_behavior_order_shared
  )
}
