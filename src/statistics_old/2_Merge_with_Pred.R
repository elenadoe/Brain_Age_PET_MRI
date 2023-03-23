######################################
#             format data            #
######################################
# DEFINE
rm(list=ls())
group <- 'MCI'
atlas <- 'Sch_Tian_1mm'
model.add <- ifelse(group == 'CN', '', '_BAGGED')

# LOAD DATA
# predictions
pred_table_pet <- read.csv(sprintf(
  '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/%s/PET-predicted_age_%s_%s%s.csv',
  group, atlas, group, model.add))
colnames(pred_table_pet)[which(names(pred_table_pet) == "Age")] <- "PET.Age"
colnames(pred_table_pet)[which(names(pred_table_pet) == "Prediction")] <- "PET.Age.pred"
pred_table_mri <- read.csv(sprintf(
  '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/%s/MRI-predicted_age_%s_%s%s.csv',
  group, atlas, group, model.add))
colnames(pred_table_mri)[which(names(pred_table_mri) == "Age")] <- "MRI.Age"
colnames(pred_table_mri)[which(names(pred_table_mri) == "Prediction")] <- "MRI.Age.pred"
# pred_table_pet <- subset(pred_table_pet, select=-c(X, Unnamed..0))
# pred_table_mri <- subset(pred_table_mri, select=-c(X, Unnamed..0))

# diagnoses table yielded from Diagnosis_Merge.R
diagnoses <- read.csv(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/Diagnoses_upto2years.csv",
  na.strings = c("", "NA"))

# Parcelation table for hippocampal volume and glucose metabolism in precuneus
mri.parcels <- read.csv(
  sprintf("2_BrainAge/Brain_Age_PET_MRI/data/ADNI/%s/ADNI_MRI_%s_%s_parcels.csv",
          group, group, atlas))
mri.parcels$Hippocampus_GMV <- (mri.parcels$Hippocampus_L + mri.parcels$Hippocampus_R)/2
mri.parcels <- subset(mri.parcels,
                      select=c("name", "Hippocampus_GMV"))
pet.parcels <- read.csv(
  sprintf("2_BrainAge/Brain_Age_PET_MRI/data/ADNI/%s/ADNI_PET_%s_%s_parcels.csv",
          group, group, atlas))
pet.parcels$Precuneus_SUVR <- (pet.parcels$Precuneus_L + pet.parcels$Precuneus_R)/2
pet.parcels <- subset(pet.parcels,
                     select=c("name", "Precuneus_SUVR"))

# merge tables
df <- merge(pred_table_pet, pred_table_mri, by = "PTID", all.x = TRUE)
df <- merge(df, diagnoses, by = "PTID", all.x = TRUE)
df <- merge(df, mri.parcels, by.x = "PTID", by.y = "name", all.x = TRUE)
df <- merge(df, pet.parcels, by.x = "PTID", by.y = "name", all.x = TRUE)

df$APOE4 <- as.factor(df$APOE4)
df$TAU <- as.numeric(df$TAU)
df$PTAU <- as.numeric(df$PTAU)
df$PTGENDER <- as.factor(ifelse(df$PTGENDER == "Female", 1, 0))
df$meanage <- rowMeans(df[,c("PET.Age", "MRI.Age")])
    
df$PET.BAG <- df$PET.Age.pred - df$PET.Age
df$MRI.BAG <- df$MRI.Age.pred - df$MRI.Age

df$AT <- ifelse(is.na(df$SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF) | is.na(df$PTAU_22_cutoff), NA,
                ifelse(df$SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF == 1 & df$PTAU_22_cutoff == 1, "A+T+",
                       ifelse(df$SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF == 1, "A+T-",
                              ifelse(df$PTAU_22_cutoff == 1, "A-T+", "A-T-"))))
table(df$AT, useNA="always")


if (group == "CN" | group == "SMC" | group == "CU") {
  # no one in CN group converted to dementia faster than after 24 months
  df$DX_final <- ifelse(!is.na(df$DX24), df$DX24,
                        ifelse(!(is.na(df$DX18)) & (df$DX18 == "MCI"), "MCI",
                        ifelse(!(is.na(df$DX12)) & (df$DX12 == "MCI"), "MCI",
                        ifelse(!(is.na(df$DX6)) & (df$DX6 == "MCI"), "MCI", NA))))
  df$DX.cat.n <- factor(ifelse(df$DX_final == "CN", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "CN", "X0_stable", "X1_decl"))
  # df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
} else {
  # remove diagnosis improvements ("CN")
  # --> only consider conversion to Dementia  
  df$DX_final <- ifelse((!is.na(df$DX24) & df$DX24 != "CN"), df$DX24,
                        ifelse(!(is.na(df$DX18)) & (df$DX18 == "Dementia"), "Dementia",
                        ifelse(!(is.na(df$DX12)) & (df$DX12 == "Dementia"), "Dementia",
                        ifelse(!(is.na(df$DX6)) & (df$DX6 == "Dementia"), "Dementia", NA))))
  table(df$DX_final)
  # df <- df[df$DX_final != "CN",]
  df$DX.cat.n <- factor(ifelse(df$DX_final == "MCI", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "MCI", "X0_stable", "X1_decl"))
  # df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
}
df$DX.bl <- group

write.csv(df, paste(sprintf("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/%s/", group),
                    sprintf("merged_for_dx_prediction_%s_%s.csv", atlas, group),
                    sep=""),
          row.names = F)
rm(list=ls())
atlas <- 'AAL1_cropped'
cn <- read.csv(paste("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/CN/",
            sprintf("merged_for_dx_prediction_%s_CN.csv", atlas), sep = ""))
smc <- read.csv(paste("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/SMC/",
            sprintf("merged_for_dx_prediction_%s_SMC.csv", atlas), sep = ""))
mci <- read.csv(paste("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/MCI/",
            sprintf("merged_for_dx_prediction_%s_MCI.csv", atlas), sep = ""))
# make sure columns are equal
cn <- cn[,(names(cn) %in% names(mci))]
all(names(cn) == names(mci))
cu <- rbind(cn, smc)
# cu <- rbind(cu, cu.init)
write.csv(cu, paste("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/",
                     sprintf("merged_for_dx_prediction_%s_CU.csv", atlas),
                     sep=""),
          row.names = F)
all <- rbind(cu, mci)
write.csv(all, paste("2_BrainAge/Brain_Age_PET_MRI/results/ADNI/",
                    sprintf("merged_for_dx_prediction_%s_all.csv", atlas),
                    sep=""),
          row.names = F)

rm(list=ls())
scd.delcode.petage <- read.csv(
  "2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/SMC/PET-predicted_age_AAL1_cropped_SMC_BAGGED.csv")
scd.dems <- read.csv(
  "2_BrainAge//Brain_Age_PET_MRI/data/DELCODE/SCD.csv")
scd <- merge(scd.delcode.petage, scd.dems, by.x = "PTID", by.y = "Repseudonym")
names(scd)[names(scd) == "ratio_Abeta42_40"] <- "ABETA42.40"
scd$PTAU.ABETA42 <- scd$phosphotau181/scd$Abeta42
scd$PET.BAG <- scd$Prediction - scd$Age
scd$APOE4 <- paste(substr(scd$ApoE, 2, 2),
                   ifelse(substr(scd$ApoE, 5, 5) == "M",
                          3, ifelse(substr(scd$ApoE, 5, 5) == "A", 4, ifelse(
                            is.na(scd$ApoE), "", 2))), sep="")
scd$APOE4 <- ifelse(scd$APOE4 == "44", 2, ifelse(grepl("4", scd$APOE4), 1, 0))
write.csv(scd, paste("2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/",
                     "merged_for_dx_prediction_AAL1_cropped_SMC.csv",
                     sep=""),
          row.names = F)

rm(list=ls())
mci.delcode.mriage <- read.csv(
  "2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/MCI/MRI-predicted_age_AAL1_cropped_MCI_BAGGED.csv")
mci.dems <- read.csv(
  "2_BrainAge//Brain_Age_PET_MRI/data/DELCODE/MCI.csv")
mci <- merge(mci.delcode.mriage, mci.dems, by.x = "PTID", by.y = "Repseudonym")
names(mci)[names(mci) == "ratio_Abeta42_40"] <- "ABETA42.40"
mci$PTAU.ABETA42 <- mci$phosphotau181/mci$Abeta42
mci$MRI.BAG <- mci$Prediction - mci$Age
mci$APOE4 <- paste(substr(mci$ApoE, 2, 2),
                   ifelse(substr(mci$ApoE, 5, 5) == "M",
                          3, ifelse(substr(mci$ApoE, 5, 5) == "A", 4, ifelse(
                            is.na(mci$ApoE), "", 2))), sep="")
mci$APOE4 <- ifelse(mci$APOE4 == "44", 2, ifelse(grepl("4", mci$APOE4), 1, 0))
write.csv(mci, paste("2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/",
                     "merged_for_dx_prediction_AAL1_cropped_MCI.csv",
                     sep=""),
          row.names = F)

