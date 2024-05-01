list.of.packages <- c("data.table", "tidyverse", "Hmisc")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)
rm(list.of.packages,new.packages)

setwd("~/git/borrowing-climate-action-africa/")

dat = fread("large_data/are_climate_unfiltered.csv")

unique_together = c(
  "crs_id", "year", "project_title", "usd_disbursement_deflated", "purpose_code"
)

dat = dat %>%
  unite(unique_id, all_of(unique_together), sep="|", na.rm=T, remove=F)

stopifnot({
  length(unique(dat$unique_id))==nrow(dat)
})

textual_cols_for_classification = c(
  "project_title",
  "short_description",
  "long_description"
)

dat = dat %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T)

keep = c(
  "unique_id",
  "text",
  "climate_adaptation",
  "climate_mitigation"
)
dat = dat[,keep]
dat = dat[complete.cases(dat),]
dat = dat[which(!duplicated(dat$text)),]
fwrite(dat,"data/uae_for_nlp.csv")
