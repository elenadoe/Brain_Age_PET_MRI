adni <- read.csv('C:/Users/doeringe/Documents/ADNImerge/ADNImerge_full.csv',
                 sep = ";", na.strings = c("NA", ""))
adni <- adni[(adni$VISCODE=="bl"),]
adni <- adni[(adni$DX == "CN"),]

rownames(adni) <- NULL

data <- read.csv2('C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/ADNI/ADNI_PET_Sch_Tian_1mm_parcels.csv')

adni <- subset(adni, select = c(PTID, ABETA, AV45))
names(adni)[names(adni) == "PTID"] <- "name"
adni <- adni[adni$name %in% data$name,]

df <- merge(data,adni,by="name",all=TRUE)

df$ABETA[df$ABETA == ">1700"] <- 1700  

amyloid_negatives <- subset(df,(df$ABETA>=192))
