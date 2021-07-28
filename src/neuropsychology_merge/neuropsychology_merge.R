adni <- read.csv('ADNImerge/ADNImerge_full.csv',
                 sep = ";", na.strings = c("NA", ""))
adni <- adni[(adni$VISCODE=="bl"),]
adni <- adni[(adni$DX == "CN"),]

rownames(adni) <- NULL

data <- read.csv('BrainAge/PET_MRI_age/data/ADNI/test_train_PET.csv')

adni <- subset(adni, select = c(PTID, MMSE, CDRSB, ADAS13, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting,
                       FAQ, MOCA, EcogPtMem, EcogPtLang, EcogPtVisspat, EcogPtPlan,
                       EcogPtOrgan, EcogPtDivatt))

adni <- na.omit(adni)
data$MMSE <- NA
data$CDRSB <- NA
data$ADAS13 <- NA
data$RAVLT.immediate <- NA
data$RAVLT.learning <- NA
data$RAVLT.forgetting <- NA
data$FAQ <- NA
data$MOCA <- NA
data$EcogPTMem <- NA
data$EcogPTLang <- NA
data$EcogPTVisspa <- NA
data$EcogPTPlan <- NA
data$EcogPTOrgan <- NA
data$EcogPTDivatt <- NA
for (i in 1:nrow(data)){
  pat <- data$name[i]
  if (pat %in% adni$PTID){
    data$MMSE[i] <- adni$MMSE[adni$PTID == pat & !is.na(adni$MMSE)]
    data$CDRSB[i] <- adni$CDRSB[adni$PTID == pat & !is.na(adni$CDRSB)]
    data$ADAS13[i] <- adni$ADAS13[adni$PTID == pat & !is.na(adni$ADAS13)]
    data$RAVLT.immediate[i] <- adni$RAVLT_immediate[adni$PTID == pat & !is.na(adni$RAVLT_immediate)]
    data$RAVLT.learning[i] <- adni$RAVLT_learning[adni$PTID == pat & !is.na(adni$RAVLT_learning)]
    data$RAVLT.forgetting[i] <- adni$RAVLT_forgetting[adni$PTID == pat & !is.na(adni$RAVLT_forgetting)]
    data$FAQ[i] <- adni$FAQ[adni$PTID == pat & !is.na(adni$FAQ)]
    data$MOCA[i] <- adni$MOCA[adni$PTID == pat & !is.na(adni$MOCA)]
    data$EcogPTMem[i] <-  adni$EcogPtMem[adni$PTID == pat & !is.na(adni$EcogPtMem)]
    data$EcogPTLang[i] <-  adni$EcogPtLang[adni$PTID == pat & !is.na(adni$EcogPtLang)]
    data$EcogPTVisspa[i] <-  adni$EcogPtVisspat[adni$PTID == pat & !is.na(adni$EcogPtVisspat)]
    data$EcogPTPlan[i] <-  adni$EcogPtPlan[adni$PTID == pat & !is.na(adni$EcogPtPlan)]
    data$EcogPTOrgan[i] <-  adni$EcogPtOrgan[adni$PTID == pat & !is.na(adni$EcogPtOrgan)]
    data$EcogPTDivatt[i] <-  adni$EcogPtDivatt[adni$PTID == pat & !is.na(adni$EcogPtDivatt)]
  }
}

write.csv(data, "BrainAge/PET_MRI_age/data/ADNI/test_train_PET_NP.csv", row.names = FALSE)
