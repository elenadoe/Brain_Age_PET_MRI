rm(list=ls())

# LOAD DATA
adnimerge <- read.csv("ADNImerge/ADNIMERGE_mostrecent.csv")
baseline <- adnimerge[adnimerge$VISCODE=="bl",]
six_months <- adnimerge[adnimerge$VISCODE=="m06",]
twelve_months <- adnimerge[adnimerge$VISCODE=="m12",]
eighteen_months <- adnimerge[adnimerge$VISCODE=="m18",]
twenty_four_months <- adnimerge[adnimerge$VISCODE=="m24",]
colnames(six_months)[which(names(six_months) == "DX")] <- "DX6"
colnames(twelve_months)[which(names(twelve_months) == "DX")] <- "DX12"
colnames(eighteen_months)[which(names(eighteen_months) == "DX")] <- "DX18"
colnames(twenty_four_months)[which(names(twenty_four_months) == "DX")] <- "DX24"

# neuropathology and neuropsychology data
neuropath_csf <- read.csv2(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/UPENNBIOMK_all.csv")
neuropath_csf <- neuropath_csf[neuropath_csf$VISCODE2=="bl",]
neuropath_csf$RUNDATE <- as.Date(neuropath_csf$RUNDATE, format="%d.%m.%Y")
# if two data points exist for a participant, only keep latest analysis
duplicates <- neuropath_csf$RID[duplicated(neuropath_csf$RID)]
for (i in 1:length(duplicates)){
  id_temp <- duplicates[i]
  df_temp <- neuropath_csf[neuropath_csf$RID == id_temp,]
  if (nrow(df_temp)>1){
    temp_max <- max(df_temp$RUNDATE)
    neuropath_csf <- neuropath_csf[-(which((neuropath_csf$RID == id_temp) &
                                     (neuropath_csf$RUNDATE != temp_max))),]
  }
}

# LOAD PET DATA
amyloid_pet <- read.csv2(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/UCBERKELEYAV45_01_14_21.csv")
amyloid_pet <- amyloid_pet[amyloid_pet$VISCODE2=="bl",]
tau_pet <- read.csv2(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/UCBERKELEYAV1451_11_16_21.csv"
)
tau_pet <- tau_pet[tau_pet$VISCODE2=="bl",]

# LOAD COGNITIVE DATA
neuropsych <- read.csv2(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/UWNPSYCHSUM_Feb2022.csv")

# MERGE DATA
# longitudinal diagnoses
df_dx <- merge(subset(baseline, select=c("PTID", "RID", "DX", "APOE4", "MMSE",
                                         "PTEDUCAT", "PTGENDER")),
               subset(six_months, select=c("PTID", "DX6")),
               by = "PTID", all.x = T, all.y = F)
df_dx <- merge(df_dx,
               subset(twelve_months, select=c("PTID", "DX12")),
               by = "PTID", all.x = T, all.y = F)
df_dx <- merge(df_dx,
               subset(eighteen_months, select=c("PTID", "DX18")),
               by = "PTID", all.x = T, all.y = F)
df_dx <- merge(df_dx, subset(twenty_four_months, select=c("PTID", "DX24")),
               by = "PTID", all.x = T, all.y = F)
# neuropathology & neuropsychology
df_dx <- merge(df_dx, subset(neuropath_csf, select=c(
  "RID", "ABETA42_recalculated", "ABETA42_cutoff", "PTAU", "TAU", "ABETA42.40", "PTAU.ABETA42",
  "PTAU.ABETA42_0.023_cutoff", "PTAU_22_cutoff")),
  by="RID", all.x = T, all.y = F)
df_dx <- merge(df_dx, subset(amyloid_pet, select=c(
  "RID", "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF", "SUMMARYSUVR_WHOLECEREBNORM")),
  by="RID", all.x = T, all.y = F)
df_dx <- merge(df_dx, subset(tau_pet, select=c(
  "RID", "meta.ROI_1.33_cutoff", "META_TEMPORAL_BY_INFCER")),
  by="RID", all.x = T, all.y = F)
df_dx <- merge(df_dx, neuropsych, by="RID", all.x = T, all.y = F)

write.csv(df_dx, "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/Diagnoses_upto2years.csv",
          row.names = F)

