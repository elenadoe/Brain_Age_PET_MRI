rm(list=ls())
amyloid_pet <- read.csv2(
  "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/PsychPath/UCBERKELEYAV45_01_14_21.csv")
amyloid_pet <- amyloid_pet[amyloid_pet$VISCODE2=="bl",]
fdg <- read.csv2("2_BrainAge/Brain_Age_PET_MRI/data/ADNI/CU/FDG_BASELINE_HEALTHY_4_15_2021_unique.csv")

fdg$RID <- substr(fdg$Subject, 7, 12)
fdg <- merge(fdg, subset(amyloid_pet,
                         select=c("RID", "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF")),
             all.x = TRUE, all.y = FALSE, by = "RID")
fdg$TRAIN <- ifelse(fdg$SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF == 0 & fdg$Group == "CN" & fdg$StudyPhase != "ADNI Baseline",
                    1, 0)
table(fdg$TRAIN)
write.csv(fdg, "2_BrainAge/Brain_Age_PET_MRI/data/ADNI/CU/FDG_BASELINE_HEALTHY_TRAINSET.csv",
          row.names = FALSE)
