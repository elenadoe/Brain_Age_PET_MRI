### DEMOGRAPHICS ###
# Compute manually for OASIS
library(table1)

rm(list=ls())
group <- "MCI"
database <- "ADNI"
atlas <- "AAL1_cropped"
modality <- "MRI"

# if CN, participants were possibly excluded by outlier exclusion
# therefore, get parcels and do new merge
if (group == "CN"){
  if (database == "ADNI"){
    # Load data
    diagnoses <- read.csv(
      "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/Diagnoses_upto2years.csv",
      na.strings = c("", "NA"))
    df <- read.csv(sprintf(
      "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/CN/ADNI_%s_CN_%s_parcels.csv",
      modality, atlas))
    cu <- read.csv2(
      "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/CU/FDG_BASELINE_HEALTHY_4_15_2021_unique.csv"
    )
    # Exclude ADNI-1 due to unknown SCD/CN
    df <- df[df$name %in% cu$Subject[cu$StudyPhase != "ADNI Baseline"],]
    # Merge with vars
    df <- merge(df, diagnoses, how = "left", by.x = "name", by.y = "PTID")
  } 
  else if (database == "OASIS"){
    diagnoses <- read.csv2(sprintf(
      "2_BrainAge/Brain_Age_PET_MRI/data/OASIS/CN/OASIS_demographics_%s.csv",
      modality), na.strings = c("", "NA"))
    dx <- read.csv2(
      "2_BrainAge/Brain_Age_PET_MRI/data/OASIS/CN/OASIS_SCDeval_USDb9.csv",
      na.strings = c("", "NA"))
    cn_ids <- dx$name[dx$group == "CN"]
    cn_ids <- cn_ids[!is.na(cn_ids)]
    # Only keep individuals who are CN as per dx table
    df <- diagnoses[diagnoses$PTID %in% cn_ids, ]
  }
} else{
  df <- read.csv(sprintf(
  "2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/merged_for_dx_prediction_%s_%s.csv",
  database, group, atlas, group))
  colnames(df)[which(names(df) == paste(modality, '.Age', sep=""))] <- "age"
}

if (database == "DELCODE"){
  df$ABETA <- ifelse(df$Abeta42>=496, 0, 1)
  colnames(df)[which(names(df) == "mmstot")] <- "MMSE"
  df$MMSE <- as.numeric(df$MMSE)
  colnames(df)[which(names(df) == "Age")] <- "age"
} else if (database == "ADNI"){
  df$ABETA <- ifelse(df$ABETA42_recalculated>=1100, 0, 1)
  df$APOE4 <- ifelse(df$APOE4>=1, 1, 0)
} else{
  df$ABETA <- NA
  df$APOE4 <- NA
  Group <- group
}


tbl1 <- table1(~ age + factor(PTGENDER) + factor(ABETA) +
                 MMSE + PTEDUCAT + factor(APOE4), data=df)
tbl1
