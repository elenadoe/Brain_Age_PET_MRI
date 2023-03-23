#########################################
#         DEMOGRAPHIC OVERVIEW          #
#########################################

rm(list=ls())
ds.stats.numerical <- function(df){
  paste(sprintf("n = %i", nrow(df)),
                sprintf("Mean age = %f, SD = %f", mean(df$Age), sd(df$Age)),
                sprintf("Mean MMSE = %f, SD = %f", mean(as.numeric(df$MMSE),
                                                        na.rm=TRUE),
                        sd(as.numeric(df$MMSE), na.rm=TRUE)),
                sprintf("Mean Education = %f, SD = %f",
                        mean(as.numeric(df$PTEDUCAT), na.rm=TRUE),
                        sd(as.numeric(df$PTEDUCAT), na.rm=TRUE)))
}
ds.stats.categorical <- function(df){
  print(table(df$AMYSTAT))
  print(table(df$PTGENDER))
}
ds.difference <- function(ds){
  print('---age---')
  for (i in 1:(length(ds)-1)){
    t <- t.test(ds[[1]]$Age, ds[[i+1]]$Age)
    print(paste(t$data.name, 't=', t$statistic,
                'p=', t$p.value))
  }
  print('---MMSE---')
  for (i in 1:(length(ds)-1)){
    t <- t.test(as.numeric(ds[[1]]$MMSE), as.numeric(ds[[i+1]]$MMSE))
    print(paste(t$data.name, 't=', t$statistic,
                'p=', t$p.value))
  }
  print('---Education---')
  for (i in 1:(length(ds)-1)){
    t <- t.test(as.numeric(ds[[1]]$PTEDUCAT), as.numeric(ds[[i+1]]$PTEDUCAT))
    print(paste(t$data.name, 't=', t$statistic,
                'p=', t$p.value))
  }
}

# Demographics tables
dem.ADNI <- read.csv2("ADNImerge/ADNIMERGE_full.csv", na.strings = "", dec=".")
dem.ADNI$ABETA <- ifelse(dem.ADNI$ABETA == ">1700", 1700, dem.ADNI$ABETA
)
dem.ADNI <- dem.ADNI[dem.ADNI$VISCODE=="bl",]
dem.ADNI$AMYSTAT <- ifelse(as.numeric(dem.ADNI$ABETA)>=1100, 0, 1)
exclude.ADNI <- read.csv2("2_BrainAge/PET_MRI_age_final/data/ADNI/exclude_below60.csv")
exclude.ADNI.CN <- exclude.ADNI$CN
exclude.ADNI.MCI <- exclude.ADNI$MCI
exclude.OASIS <- 'OAS31168'

### CN + SCD ###
modality <- 'PET'
dem.OASIS <- read.csv2(sprintf(
  "2_BrainAge/PET_MRI_age_final/data/OASIS/OASIS_demographics_%s.csv", modality))
dem.OASIS <- subset(dem.OASIS, select = -c(Age))
dem.DELCODE <- read.csv2(
  "2_BrainAge/PET_MRI_age_final/data/DELCODE/SCD/SCD.csv", na.strings = ""
)
dem.DELCODE$PTGENDER <- ifelse(dem.DELCODE$sex == "m", "Male", "Female")
dem.DELCODE$AMYSTAT <- ifelse(dem.DELCODE$ABETA>=496, 0, 1)
dem.DELCODE <- subset(dem.DELCODE, select = -c(Age))
participants.ADNI <- read.csv2(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/CN/%s_parcels_CN_ADNI.csv', modality))
colnames(participants.ADNI)[which(names(participants.ADNI)=='age')] <- 'Age'
participants.ADNI <- participants.ADNI[!(participants.ADNI$name %in% exclude.ADNI.CN),]

participants.OASIS <- read.csv2(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/OASIS/OASIS_parcels_%s.csv', modality), dec=".")
participants.OASIS <- participants.OASIS[participants.OASIS$Age >= 60,]
participants.OASIS <- participants.OASIS[participants.OASIS$name != exclude.OASIS,]

participants.DELCODE <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/DELCODE/SCD/%s_parcels_SCD_DELCODE.csv',
  modality))
colnames(participants.DELCODE)[which(names(participants.DELCODE)=='age')] <- 'Age'
participants.DELCODE <- participants.DELCODE[participants.DELCODE$Age >= 60,]

# merge
participants.ADNI <- merge(participants.ADNI, dem.ADNI, by.x='name', by.y='PTID',
                           all.x=TRUE)
table(participants.ADNI$DX_bl)
participants.OASIS <- merge(participants.OASIS, dem.OASIS, by.x='name',
                            by.y='PTID', all.x=TRUE)
participants.DELCODE <- merge(participants.DELCODE, dem.DELCODE, by.x='name',
                              by.y='PTID', all.x=TRUE)


ds <- list(participants.ADNI, participants.OASIS, participants.DELCODE)
lapply(ds, ds.stats.numerical)
lapply(ds, ds.stats.categorical)
ds.difference(ds)

# clear variable space
rm(participants.ADNI)
rm(participants.OASIS)
rm(participants.DELCODE)

# assess MRI
modality <- 'MRI'
dem.OASIS <- read.csv2(sprintf(
  "2_BrainAge/PET_MRI_age_final/data/OASIS/OASIS_demographics_%s.csv", modality),
  na.strings = "")
dem.OASIS <- subset(dem.OASIS, select = -c(Age))
participants.ADNI <- read.csv2(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/CN/%s_parcels_CN_ADNI.csv', modality))
colnames(participants.ADNI)[which(names(participants.ADNI)=='age')] <- 'Age'
participants.ADNI <- participants.ADNI[!(participants.ADNI$name %in% exclude.ADNI.CN),]

participants.OASIS <- read.csv2(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/OASIS/OASIS_parcels_%s.csv', modality))
colnames(participants.OASIS)[which(names(participants.OASIS)=='age')] <- 'Age'
participants.OASIS <- participants.OASIS[participants.OASIS$Age >= 60,]
participants.OASIS <- participants.OASIS[participants.OASIS$name != exclude.OASIS,]

participants.ADNI <- merge(participants.ADNI, dem.ADNI, by.x='name', by.y='PTID',
                           all.x=TRUE)
participants.OASIS <- merge(participants.OASIS, dem.OASIS, by.x='name',
                            by.y='PTID', all.x=TRUE)

ds <- list(participants.ADNI, participants.OASIS)
lapply(ds, ds.stats.numerical)
lapply(ds, ds.stats.categorical)
ds.difference(ds)

# clear variable space
rm(participants.ADNI)
rm(participants.OASIS)

# MCI
# TODO compile again starting here
modality <- 'PET'
participants.ADNI <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/MCI/%s_parcels_MCI_ADNI.csv', modality))
participants.ADNI <- participants.ADNI[!(participants.ADNI$name %in% exclude.ADNI.MCI),]
colnames(participants.ADNI)[which(names(participants.ADNI)=='age')] <- 'Age'
participants.ADNI <- participants.ADNI[participants.ADNI$Age >= 60,]

participants.ADNI <- merge(participants.ADNI, dem.ADNI, by.x='name', by.y='PTID',
                           all.x=TRUE)
table(participants.ADNI$DX_bl)
ds <- list(participants.ADNI)
lapply(ds, ds.stats.numerical)
lapply(ds, ds.stats.categorical)

# clear variable space
rm(participants.ADNI)

modality <- 'MRI'
dem.DELCODE <- read.csv2(
  "2_BrainAge/PET_MRI_age_final/data/DELCODE/MCI/MCI.csv", na.strings = "",
)
dem.DELCODE$PTGENDER <- ifelse(dem.DELCODE$sex == "m", "Male", "Female")
dem.DELCODE$AMYSTAT <- ifelse(dem.DELCODE$ABETA>=496, 0, 1)
dem.DELCODE <- subset(dem.DELCODE, select = -c(Age))
participants.ADNI <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/MCI/%s_parcels_MCI_ADNI.csv', modality))
colnames(participants.ADNI)[which(names(participants.ADNI)=='age')] <- 'Age'
participants.ADNI <- participants.ADNI[!(participants.ADNI$name %in% exclude.ADNI.MCI),]
participants.ADNI <- participants.ADNI[participants.ADNI$Age >= 60,]
participants.DELCODE <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/DELCODE/MCI/%s_parcels_MCI_DELCODE.csv',
  modality))
colnames(participants.DELCODE)[which(names(participants.DELCODE)=='age')] <- 'Age'
participants.DELCODE <- participants.DELCODE[participants.DELCODE$Age >= 60,]

participants.ADNI <- merge(participants.ADNI, dem.ADNI, by.x='name', by.y='PTID',
                           all.x=TRUE)
participants.DELCODE <- merge(participants.DELCODE, dem.DELCODE, by.x='name',
                            by.y='PTID', all.x=TRUE)

ds <- list(participants.ADNI, participants.DELCODE)
lapply(ds, ds.stats.numerical)
lapply(ds, ds.stats.categorical)
ds.difference(ds)

### REFERENCES ###
# DELCODE CSF CUTOFF: https://alzres.biomedcentral.com/articles/10.1186/s13195-017-0314-2
