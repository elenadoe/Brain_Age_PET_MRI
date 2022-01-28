adni <- read.csv('C:/Users/doeringe/Documents/ADNImerge/ADNImerge_full2.csv',
                 sep = ";", na.strings = c("NA", ""), stringsAsFactors = FALSE)
adni <- adni[(adni$VISCODE=="bl"),]
adni <- adni[(adni$DX == 'MCI'),]

rownames(adni) <- NULL

data <- read.csv2('C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/ADNI/MCI_PET_parcels.csv',
                 stringsAsFactors = FALSE)

adni <- subset(adni, select = c(name, PTGENDER, MMSE, ABETA, PTAU, TAU, AV45, CDRSB, ADAS13, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting,
                       FAQ, MOCA, EcogPtMem, EcogPtLang, EcogPtVisspat, EcogPtPlan,
                       EcogPtOrgan, EcogPtDivatt))

df <- merge(data,adni,by='name',all.x=TRUE)

df$ABETA[df$ABETA == ">1700"] <- 1700
df$PTAU[df$TAU == '<8'] <- 8
df$TAU[df$TAU == '<80'] <- 80

write.csv(df, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/ADNI/MCI_PET_parcels_with_NP.csv", row.names = FALSE)
