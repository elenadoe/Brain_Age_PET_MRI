### FOR STATISTICS
rm(list=ls())
group <- "MCI"
diagnosis_bl <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_bl2.csv",
                                na.strings = "")
if (group == "CN"){
  indiv <- read.csv("2_BrainAge/PET_MRI_age/data/main/test_train_MRI_0.csv")
  indiv <- indiv[indiv$Dataset=="ADNI",]
  indiv <- indiv[indiv$AGE_CHECK=="True",]
}else{
  indiv <- read.csv2("2_BrainAge/PET_MRI_age/data/MCI/MCI_MRI_parcels.csv")
  indiv <- indiv[indiv$age >= 65,]
}


#diagnosis_bl$name <- diagnosis_bl$PTID
df <- merge(indiv, diagnosis_bl, by="name", all.x=TRUE, all.y=FALSE)

df <- read.csv2("2_BrainAge/PET_MRI_age/data/main/OASIS_demographics_MRI.csv",
                          na.strings = "")

df$AMYCAT <- ifelse(as.numeric(df$ABETA)<1100, 1, 0)
table(df$AMYCAT, useNA = "always")
df$AMYCAT2 <- ifelse(as.numeric(df$AV45)>1.11, 1, 0)
table(df$AMYCAT2, useNA = "always")
table(df$PTGENDER)
table(df$PTRACCAT)
mean(df$MMSE, na.rm = TRUE)
sd(df$MMSE, na.rm = TRUE)
mean(df$PTEDUCAT, na.rm = TRUE)
sd(df$PTEDUCAT, na.rm = TRUE)
