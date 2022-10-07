######################################
#             format data            #
######################################

# DEFINE
rm(list=ls())
group <- 'CN'
model.add <- ifelse(group == 'CN', '', '_BAGGED')

# LOAD DATA
# predictions
pred_table_pet <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/results/ADNI/%s/PET-predicted_age_%s%s.csv',
  group, group, model.add))
colnames(pred_table_pet)[which(names(pred_table_pet) == "Age")] <- "PET.Age"
colnames(pred_table_pet)[which(names(pred_table_pet) == "Prediction")] <- "PET.Age.pred"
pred_table_mri <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/results/ADNI/%s/MRI-predicted_age_%s%s.csv',
  group, group, model.add))
colnames(pred_table_mri)[which(names(pred_table_mri) == "Age")] <- "MRI.Age"
colnames(pred_table_mri)[which(names(pred_table_mri) == "Prediction")] <- "MRI.Age.pred"
# pred_table_pet <- subset(pred_table_pet, select=-c(X, Unnamed..0))
# pred_table_mri <- subset(pred_table_mri, select=-c(X, Unnamed..0))

# diagnoses table yielded from Diagnosis_Merge.R
diagnoses <- read.csv(
  "2_BrainAge/PET_MRI_age_final/data/ADNI/PsychPath/Diagnoses_upto2years.csv",
  na.strings = c("", "NA"))

# merge tables
df <- merge(pred_table_pet, pred_table_mri, by = "PTID", all.x = TRUE)
df <- merge(df, diagnoses, by = "PTID", all.x = TRUE)

df$APOE4 <- factor(ifelse(df$APOE4==0, 1, 2))
df$PTGENDER <- factor(ifelse(df$PTGENDER == "Female", 1, 0))
df$meanage <- rowMeans(df[,c("PET.Age", "MRI.Age")])
df$ABETA <- as.numeric(df$ABETA)
df$ABETA.cat <- factor(ifelse(is.na(df$ABETA), 0, ifelse(df$ABETA>=1100, -1, 1)))
df$PET.BAG <- df$PET.Age.pred - df$PET.Age
df$MRI.BAG <- df$MRI.Age.pred - df$MRI.Age

if (group == "CN") {
  # no one in CN group converted to dementia faster than after 24 months
  df$DX_final <- ifelse(!is.na(df$DX24), df$DX24,
                        ifelse(!(is.na(df$DX18)) & (df$DX18 == "MCI"), "MCI",
                        ifelse(!(is.na(df$DX12)) & (df$DX12 == "MCI"), "MCI",
                        ifelse(!(is.na(df$DX6)) & (df$DX6 == "MCI"), "MCI", NA))))
  df$DX.cat.n <- factor(ifelse(df$DX_final == "CN", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "CN", "X0_stable", "X1_decl"))
  df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
} else {
  # remove diagnosis improvements ("CN")
  # --> only consider conversion to Dementia  
  df$DX_final <- ifelse(!is.na(df$DX24), df$DX24,
                        ifelse(!(is.na(df$DX18)) & (df$DX18 == "Dementia"), "Dementia",
                        ifelse(!(is.na(df$DX12)) & (df$DX12 == "Dementia"), "Dementia",
                        ifelse(!(is.na(df$DX6)) & (df$DX6 == "Dementia"), "Dementia", NA))))
  table(df$DX_final)
  # df <- df[df$DX_final != "CN",]
  df$DX.cat.n <- factor(ifelse(df$DX_final == "MCI", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "MCI", "X0_stable", "X1_decl"))
  df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
  
}

write.csv(df, paste("2_BrainAge/PET_MRI_age_final/data/ADNI/PsychPath/",
                    sprintf("merged_for_dx_prediction_%s.csv", group),
                    sep=""),
          row.names = F)
