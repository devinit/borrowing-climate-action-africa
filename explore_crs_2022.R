list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

YEAR = 2022

wd = "~/git/borrowing-climate-action-africa/"
setwd(wd)

crs = fread(paste0("large_data/crs_",YEAR,"_predictions.csv"))
original_names = names(crs)[1:95]

# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Climate adaptation - significant objective confidence`[blank_indices] = 0
crs$`Climate adaptation - significant objective predicted`[blank_indices] = F
crs$`Climate adaptation - principal objective confidence`[blank_indices] = 0
crs$`Climate adaptation - principal objective predicted`[blank_indices] = F
crs$`Climate mitigation - significant objective confidence`[blank_indices] = 0
crs$`Climate mitigation - significant objective predicted`[blank_indices] = F
crs$`Climate mitigation - principal objective confidence`[blank_indices] = 0
crs$`Climate mitigation - principal objective predicted`[blank_indices] = F

crs$`Principal climate adaptation` = F
crs$`Principal climate adaptation`[which(crs$climate_adaptation==2)] = T
crs$`Principal climate adaptation`[which(
  crs$`Climate adaptation - principal objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Principal climate mitigation` = F
crs$`Principal climate mitigation`[which(crs$climate_mitigation==2)] = T
crs$`Principal climate mitigation`[which(
  crs$`Climate mitigation - principal objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Significant climate adaptation` = F
crs$`Significant climate adaptation`[which(crs$climate_adaptation==1)] = T
crs$`Significant climate adaptation`[which(
  crs$`Climate adaptation - significant objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Significant climate mitigation` = F
crs$`Significant climate mitigation`[which(crs$climate_mitigation==1)] = T
crs$`Significant climate mitigation`[which(
  crs$`Climate mitigation - significant objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Climate label` = ifelse(
  (crs$`Principal climate adaptation` & crs$`Principal climate mitigation`) 
  # | (crs$`Significant climate adaptation` & crs$`Principal climate mitigation`)
  # | (crs$`Principal climate adaptation` & crs$`Significant climate mitigation`)
  ,"Both",
  ifelse(
    crs$`Principal climate adaptation`,
    "Adaptation",
    ifelse(
      crs$`Principal climate mitigation`,
      "Mitigation",
      "None"
    )
  )
)

describe(crs$`Climate label`)

check_a = subset(crs, `Climate adaptation - principal objective confidence` > 0.9 & !`Climate keyword match`)
check_m = subset(crs, `Climate mitigation - principal objective confidence` > 0.9 & !`Climate keyword match`)

check_rev = subset(crs, 
                   `Climate adaptation - principal objective confidence` < 0.1 &
                     `Climate mitigation - principal objective confidence` < 0.1 &
                     `Climate keyword match`
                  )

keep = c(original_names,
        "Principal climate adaptation",
        "Principal climate mitigation",
        "Significant climate adaptation",
        "Significant climate mitigation",
        "Climate label"
)

out_crs = subset(crs, `Climate label`!="None",select=keep)
out_crs_wb = subset(crs, donor_name == 'International Development Association')

fwrite(out_crs,
       paste0("data/crs_",YEAR,"_automated.csv"))

fwrite(out_crs_wb, paste0("data/crs_wb_",YEAR,"_automated.csv"))
