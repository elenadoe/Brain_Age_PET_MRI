#######################################
###      CORRELATION WITH CI/NP     ###
#######################################
library(ggplot2)
library(tidyverse)
library(dplyr)

# load data
rm(list=ls())
clin_data_bl <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_bl.csv",
                          na.strings = "", dec = ".")
diagnosis_24months <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_24months2.csv",
                                na.strings = "")
cognition <- read.csv2(
  "2_BrainAge/PET_MRI_age/data/main/UWNPSYCHSUM_Feb2022.csv",
  na.strings = "", dec = ".")

clin_data_bl$PTID <- clin_data_bl$name
clin_data_bl$RID <- as.numeric(substr(clin_data_bl$PTID, 7, 10))
clin_data_bl <- subset(clin_data_bl, select = -c(DX))
diagnosis_24months <- subset(diagnosis_24months, select = c(PTID, DX))

clin_data <- merge(clin_data_bl, diagnosis_24months, by = "PTID", all.x = TRUE)
clin_data <- merge(clin_data_bl, cognition, by = "RID", all.x = TRUE)

group <- "MCI"
modality <- "PET"

model <- ifelse(group == "MCI", "_0", "")
pred_age_pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/", group, "/",
                               modality, "-predicted_age_", group, model, ".csv",
                               sep = ""))
modality <- "MRI"
pred_age_mri <- read.csv(paste("2_BrainAge/PET_MRI_age/results/", group, "/",
                               modality, "-predicted_age_", group, model, ".csv",
                               sep = ""))

# get age at individual scans (may vary by 1 year), BPAD and BPAD category
# for both modalities
pred_age_pet$PETBPAD <- pred_age_pet$Prediction - pred_age_pet$Age
pred_age_mri$MRIBPAD <- pred_age_mri$Prediction - pred_age_mri$Age
pred_age_pet$PETAGE <- pred_age_pet$Age
pred_age_mri$MRIAGE <- pred_age_mri$Age
pred_age_pet <- subset(pred_age_pet, select=c(PTID, PETAGE, PETBPAD))
pred_age_mri <- subset(pred_age_mri, select=c(PTID, MRIAGE, MRIBPAD))

# merge
df <- merge(pred_age_pet, pred_age_mri, by='PTID', all.x=TRUE)
df <- merge(df, clin_data, by='PTID', all.x=TRUE)

#  CORRELATION WITH NP
tiff("test.tiff", units="in", width=5, height=5, res=300)

var_of_interest <- "ADNI_EF"
# var_of_interest <- expression("\n"~CSF~A~beta["1-42"]~"\n")
g <- ggplot(df) +
  #geom_point(aes(x=PETBPAD, y=ADNI_EF, shape=PTGENDER), color="coral", fill="coral",
  #           alpha=0.2, size=3) + 
  geom_point(aes(x=MRIBPAD, y=ADNI_EF, shape=PTGENDER), color="cyan4", fill="cyan4",
             alpha=0.2, size=3) + 
  #stat_smooth(method=lm, fullrange=FALSE, aes(x=PETBPAD, y=ADNI_MEM),
  #            linetype="dashed", color="coral", fill="coral", alpha=0.3, lwd=1.2)+
  stat_smooth(method=lm, fullrange=FALSE, aes(x=MRIBPAD, y=ADNI_MEM),
              linetype="dashed", color="cyan4", fill="cyan4", alpha=0.2, lwd=1.2)+
  scale_x_continuous(breaks=seq(-12,10,4)) +
  ylab(var_of_interest) +
  xlab("\nBPAD [years]") +
  theme_classic() +
  scale_color_discrete() +
  theme(text = element_text(size=20))
g

ggsave(filename = paste(group, var_of_interest, ".png"),
       path = "2_BrainAge/PET_MRI_age/results/Figures/overlay_bothmodalities/",
       width = 10, height = 10, device='tiff', dpi=300)
