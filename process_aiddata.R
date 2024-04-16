list.of.packages <- c("data.table", "openxlsx", "tidyverse", "Hmisc")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)
rm(list.of.packages,new.packages)

setwd("~/git/borrowing-climate-action-africa/")

gcdf = read.xlsx(
  "large_data/AidDatas_Global_Chinese_Development_Finance_Dataset_Version_3_0/AidDatas_Global_Chinese_Development_Finance_Dataset_Version_3_0/AidDatasGlobalChineseDevelopmentFinanceDataset_v3.0.xlsx",
  sheet="GCDF_3.0"
)

au = fread("data/au_countries.csv")
gcdf = subset(gcdf, (`Recipient.ISO-3` %in% au$iso3) | `Recipient.Region` == "Africa")

missing_origin_date_cols = c(
  "Planned.Implementation.Start.Date.(MM/DD/YYYY)",
  "Actual.Implementation.Start.Date.(MM/DD/YYYY)",
  "Planned.Completion.Date.(MM/DD/YYYY)",
  "Actual.Completion.Date.(MM/DD/YYYY)",
  "First.Loan.Repayment.Date",
  "Last.Loan.Repayment.Date"
)
for(date_col in missing_origin_date_cols){
  gcdf[,date_col] = convertToDateTime(gcdf[,date_col], origin = "1900-01-01")
  year_date_col = paste0(date_col,".Year")
  gcdf[,year_date_col] = year(gcdf[,date_col])
}

all_year_cols = c(
  "Implementation.Start.Year",                      
  "Completion.Year",
  paste0(missing_origin_date_cols,".Year")
)

gcdf$year_min = apply(gcdf[,all_year_cols], MARGIN=1, FUN=min, na.rm=T)
gcdf$year_max = apply(gcdf[,all_year_cols], MARGIN=1, FUN=max, na.rm=T)
View(gcdf[,c(all_year_cols, "year_min", "year_max")])
gcdf$crosses_2022 = gcdf$year_min == 2022 |
  gcdf$year_max == 2022 | (
    gcdf$year_min < 2022 &
      gcdf$year_max > 2022
  )

gcdf = subset(gcdf, crosses_2022 & Flow.Type=="Loan")

textual_cols_for_classification = c(
  "Title",
  "Description",
  "Staff.Comments"
)

gcdf = gcdf %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T)

keep = c("AidData.Record.ID", "text")
gcdf = gcdf[,keep]
fwrite(gcdf, "data/filtered_gcdf_au_loan_2022.csv")
