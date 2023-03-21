##################################
#      Correlations of BAG       #
#          in CN & MCI           #
#################################

cn.pet <- read.csv('2_BrainAge/PET_MRI_age_final/results/ADNI/CN/PET-predicted_age_CN.csv')
cn.mri <- read.csv('2_BrainAge/PET_MRI_age_final/results/ADNI/CN/MRI-predicted_age_CN.csv')
mci.pet <- read.csv('2_BrainAge/PET_MRI_age_final/results/ADNI/MCI/PET-predicted_age_MCI_BAGGED.csv')
mci.mri <- read.csv('2_BrainAge/PET_MRI_age_final/results/ADNI/MCI/MRI-predicted_age_MCI_BAGGED.csv')
cn.pet$BAG <- cn.pet$Prediction - cn.pet$Age
cn.mri$BAG <- cn.mri$Prediction - cn.mri$Age
mci.pet$BAG <- mci.pet$Prediction - mci.pet$Age
mci.mri$BAG <- mci.mri$Prediction - mci.mri$Age

shapiro.test(cn.pet$BAG)  # normally distributed
shapiro.test(cn.mri$BAG)  # normally distributed
shapiro.test(mci.pet$BAG)  # not normally distributed
shapiro.test(mci.mri$BAG)    # normally distributed

cor.test(cn.pet$BAG, cn.mri$BAG, method = 'pearson')
cor.test(mci.pet$BAG, mci.mri$BAG, method = 'spearman')
