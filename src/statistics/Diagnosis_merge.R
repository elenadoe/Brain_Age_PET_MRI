rm(list=ls())
diagnosis_24months <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_24months2.csv",
                                na.strings = "")
diagnosis_12months <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_12months.csv",
                                na.strings = "")
diagnosis_6months <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_6months.csv",
                                na.strings = "")
group <- "CN"
model <- ifelse(group == "MCI", "_0", "")
pred_age_pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/", group, "/PET",
                               "-predicted_age_", group, model, ".csv", sep = ""))
diagnosis_24months <- subset(diagnosis_24months, select=c(PTID, DX))
diagnosis_12months <- subset(diagnosis_12months, select=c(PTID, DX))
diagnosis_6months <- subset(diagnosis_6months, select=c(PTID, DX))
colnames(diagnosis_6months)[which(names(diagnosis_6months) == "DX")] <- "DX6"
colnames(diagnosis_12months)[which(names(diagnosis_12months) == "DX")] <- "DX12"
colnames(diagnosis_24months)[which(names(diagnosis_24months) == "DX")] <- "DX24"

df <- merge(pred_age_pet, diagnosis_24months, by = "PTID", all.x = T)
df_dx <- merge(df, diagnosis_12months, by = "PTID", all.x = T)
df_dx <- merge(df_dx, diagnosis_6months, by = "PTID", all.x = T)
df_dx <- subset(df_dx, select = -c(Unnamed..0, X, VISCODE.x, DX_bl.x))
table(df_dx$DX6)
table(df_dx$DX12)
table(df_dx$DX24)

write.csv(df_dx, paste("2_BrainAge/PET_MRI_age/data/Diagnoses_upto2years_",
                        group, ".csv", sep = ""),
          row.names = F)
