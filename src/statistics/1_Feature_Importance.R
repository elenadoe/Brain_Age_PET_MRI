####################################
# Inspection of feature importance #
####################################
library(Hmisc)

rm(list=ls())
coef_mri_0 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_MRI_svm_0.csv")
coef_mri_1 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_MRI_svm_1.csv")
coef_mri_2 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_MRI_svm_2.csv")
coef_mri_3 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_MRI_svm_3.csv")
coef_mri_4 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_MRI_svm_4.csv")

coef_pet_0 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_PET_svm_0.csv")
coef_pet_2 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_PET_svm_2.csv")
coef_pet_3 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_PET_svm_3.csv")
coef_pet_4 <- read.csv(
  "2_BrainAge/PET_MRI_age_final/results/ADNI/CN/evaluation/weighted_importance_PET_svm_4.csv")

coef <- merge(coef_mri_0, merge(coef_mri_1, merge(coef_mri_2,
              merge(coef_mri_3, merge(coef_mri_4,
              merge(coef_pet_0, merge(coef_pet_2, merge(coef_pet_3,
              coef_pet_4, by="region"), by="region"), by="region"),
              by="region"), by="region"), by="region"), by="region"),
              by="region")
coef_mri <- merge(coef_mri_0, merge(coef_mri_1, merge(coef_mri_2,
      merge(coef_mri_3, coef_mri_4, by="region"), by="region"),
      by="region"), by="region")
coef_pet <- merge(coef_pet_0, merge(coef_pet_2, merge(coef_pet_3,
            coef_pet_4, by="region"), by="region"), by="region")

coef_mri <- subset(coef_mri, select = -c(region, X, X.x, X.y, X.x.1, X.y.1))
coef_pet <- subset(coef_pet, select = -c(region, X.x, X.y))
coef_pet <- subset(coef_pet, select = -c(X.x, X.y))

result_mri <- rcorr(as.matrix(coef_mri))
result_pet <- rcorr(as.matrix(coef_pet))

coef_mri$avg <- rowMeans(coef_mri)
coef_pet$avg <- rowMeans(coef_pet)

cor.test(coef_mri$avg, coef_pet$avg)
