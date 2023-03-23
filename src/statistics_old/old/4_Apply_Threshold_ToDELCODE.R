#######################################
###     Validation of DX change     ### 
###      prediction threshold       ###
###         with DELCODE            ###
#######################################
library(pROC)
# TODO: derive threshold from only SCD data
# maybe even take SCD out of CN cohort all together if results remain the same
rm(list=ls())
scd.pet.age <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/DELCODE/SCD/PET-predicted_age_SCD_BAGGED.csv")
scd.dxchange <- read.csv2(
  "2_BrainAge/PET_MRI_age_final/data/DELCODE/Follow-Up_MCIDX.csv"
)
scd.dxchange$MCITime_untilApril21 <- as.numeric(scd.dxchange$MCITime_untilApril21)

scd.dxchange$DX_final <- ifelse(
  # if no conversion, but observation period shorter than 1.5 years, discard
  scd.dxchange$MCI_untilApril21 == 0 &
    scd.dxchange$MCITime_untilApril21 < 1.5, NA,
  # if conversion, but only after >= 2.5 years, consider as stable until year 2
  ifelse(
    scd.dxchange$MCI_untilApril21 == 1 &
      scd.dxchange$MCITime_untilApril21 >= 2.5, "X0_stable",
  # if conversion within two years, consider as decliner
  ifelse(scd.dxchange$MCI_untilApril21 == 1, "X1_decl",
  # if no conversion, but observation period at least 2 years, consider as stable    
  "X0_stable")))

scd <- merge(scd.pet.age, scd.dxchange, by.x = 'PTID', by.y = 'Repseudonym',
             all.x = TRUE)
scd <- scd[!(is.na(scd$DX_final)),]
table(scd$DX_final)

scd$BAG <- scd$Prediction - scd$Age
scd$threshold <- ifelse(scd$BAG>0.7829160, "Y1_decl", "Y0_stable")
table(scd$DX_final, scd$threshold)

rm(list=ls())
mci.mri.age <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/DELCODE/MCI/MRI-predicted_age_MCI_BAGGED.csv")
mci.dxchange <- read.csv2(
  "2_BrainAge/PET_MRI_age (Kopie)/data/DELCODE/Follow-Up_ADDX.csv"
)
mci.dxchange$DemTime_untilApril21 <- as.numeric(mci.dxchange$DemTime_untilApril21)

mci.dxchange$DX_final <- ifelse(
  # if no conversion, but observation period shorter than 1.5 years, discard
  mci.dxchange$Dem_untilApril21 == 0 &
    mci.dxchange$DemTime_untilApril21 < 1.5, NA,
  # if conversion, but only after >= 2.5 years, consider as stable until year 2
  ifelse(
    mci.dxchange$Dem_untilApril21 == 1 &
      mci.dxchange$DemTime_untilApril21 >= 2.5, "X0_stable",
    # if conversion within two years, consider as decliner
    ifelse(mci.dxchange$Dem_untilApril21 == 1, "X1_decl",
           # if no conversion, but observation period at least 2 years, consider as stable    
           "X0_stable")))

mci <- merge(mci.mri.age, mci.dxchange, by.x = 'PTID', by.y = 'Repseudonym',
             all.x = TRUE)
mci <- mci[!(is.na(mci$DX_final)),]
table(mci$DX_final)

mci$BAG <- mci$Prediction - mci$Age

mci$threshold <- ifelse(mci$BAG>2.226893, "Y1_decl", "Y0_stable")
table(mci$DX_final, mci$threshold)
