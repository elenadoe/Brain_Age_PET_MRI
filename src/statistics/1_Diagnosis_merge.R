rm(list=ls())

# LOAD DATA
adnimerge <- read.csv2("ADNImerge/ADNIMERGE_full.csv")
baseline <- adnimerge[adnimerge$VISCODE=="bl",]
six_months <- adnimerge[adnimerge$VISCODE=="m06",]
twelve_months <- adnimerge[adnimerge$VISCODE=="m12",]
eighteen_months <- adnimerge[adnimerge$VISCODE=="m18",]
twenty_four_months <- adnimerge[adnimerge$VISCODE=="m24",]
colnames(six_months)[which(names(six_months) == "DX")] <- "DX6"
colnames(twelve_months)[which(names(twelve_months) == "DX")] <- "DX12"
colnames(eighteen_months)[which(names(eighteen_months) == "DX")] <- "DX18"
colnames(twenty_four_months)[which(names(twenty_four_months) == "DX")] <- "DX24"

# MERGE DATAs
df_dx <- merge(subset(baseline, select=c("PTID", "DX", "ABETA", "AV45",
                                         "APOE4", "PTEDUCAT", "PTGENDER")),
               subset(six_months, select=c("PTID", "DX6")),
               by = "PTID", all.x = T)
df_dx <- merge(df_dx,
               subset(twelve_months, select=c("PTID", "DX12")),
               by = "PTID", all.x = T)
df_dx <- merge(df_dx,
               subset(eighteen_months, select=c("PTID", "DX18")),
               by = "PTID", all.x = T)
df_dx <- merge(df_dx, subset(twenty_four_months, select=c("PTID", "DX24")),
               by = "PTID", all.x = T)

# replace ends of spectrum with min and max
df_dx$ABETA <- ifelse(df_dx$ABETA == "<200", 200, ifelse(df_dx$ABETA == ">1700",
                                                         1700, df_dx$ABETA))

write.csv(df_dx, "2_BrainAge/PET_MRI_age_final/data/ADNI/PsychPath/Diagnoses_upto2years.csv",
          row.names = F)
