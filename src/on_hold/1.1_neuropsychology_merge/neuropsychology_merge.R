rm(list=ls())
adni <- read.csv('C:/Users/doeringe/Documents/ADNImerge/ADNImerge_full.csv',
                 sep = ";", na.strings = c("NA", ""))
adni <- adni[(adni$VISCODE=="bl"),]
adni <- adni[(adni$DX == "CN"),]

rownames(adni) <- NULL

data <- read.csv('C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/merged/test_train_MRI_final.csv')

adni <- subset(adni, select = c(PTID, MMSE, ABETA, PTAU, TAU, AV45, CDRSB, ADAS13, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting,
                       FAQ, MOCA, EcogPtMem, EcogPtLang, EcogPtVisspat, EcogPtPlan,
                       EcogPtOrgan, EcogPtDivatt))
names(adni)[names(adni) == "PTID"] <- "name"

#adni <- na.omit(adni)
df <- merge(data,adni,all.x=TRUE,all.y=FALSE)

df$ABETA[df$ABETA == ">1700"] <- 1700
df$TAU[df$TAU == '<80'] <- 80
df$TAU[df$PTAU == '<8'] <- 8

write.csv(df, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/merged/test_train_MRI_final_withNP.csv", row.names = FALSE)
