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
  (crs$`Principal climate adaptation` & crs$`Principal climate mitigation`) |
    (crs$`Significant climate adaptation` & crs$`Principal climate mitigation`) |
    (crs$`Principal climate adaptation` & crs$`Significant climate mitigation`),
  "Both",
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

keep= c(original_names,
        "humanitarian",
        "Crisis finance identified",
        "Crisis finance eligible",
        "Crisis finance determination",
        "Crisis finance keyword match",
        "Crisis finance predicted ML",
        "Crisis finance confidence ML",
        "PAF determination",
        "PAF keyword match",
        "PAF predicted ML",
        "PAF confidence ML",
        "AA determination",
        "AA keyword match",
        "AA predicted ML",
        "AA confidence ML",
        "Direct predicted ML",
        "Direct confidence ML",
        "Indirect predicted ML",
        "Indirect confidence ML",
        "Part predicted ML",
        "Part confidence ML"
)

crs = crs[order(
  crs$`Crisis finance determination`=="No",
  crs$`Crisis finance determination`=="Review",
  crs$`Crisis finance determination`=="Yes",
  -crs$`Crisis finance confidence ML`
),keep]


fwrite(crs,
       paste0("large_data/crs_",YEAR,"_cdp_automated.csv"))
