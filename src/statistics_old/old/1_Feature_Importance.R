####################################
# Inspection of feature importance #
####################################
library(Hmisc)
library(ggplot2)
library(ggnewscale)

rm(list=ls())
coef_mri_0 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_MRI_Sch_Tian_svm_0.csv")
coef_mri_1 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_MRI_Sch_Tian_svm_1.csv")
coef_mri_2 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_MRI_Sch_Tian_svm_2.csv")
coef_mri_3 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_MRI_Sch_Tian_svm_3.csv")
coef_mri_4 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_MRI_Sch_Tian_svm_4.csv")

coef_pet_0 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_PET_Sch_Tian_svm_0.csv")
coef_pet_1 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_PET_Sch_Tian_svm_1.csv")
coef_pet_2 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_PET_Sch_Tian_svm_2.csv")
coef_pet_3 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_PET_Sch_Tian_svm_3.csv")
coef_pet_4 <- read.csv(
  "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation/weighted_importance_PET_Sch_Tian_svm_4.csv")

coef_mri <- merge(coef_mri_0, merge(coef_mri_1, merge(coef_mri_2,
      merge(coef_mri_3, coef_mri_4, by="region"), by="region"),
      by="region"), by="region")
coef_pet <- merge(coef_pet_0, merge(coef_pet_1, merge(coef_pet_2,
      merge(coef_pet_3, coef_pet_4, by="region"), by="region"),
      by="region"), by="region")

coef_mri <- subset(coef_mri, select = -c(region, X, X.x, X.y, X.x.1, X.y.1))
coef_pet <- subset(coef_pet, select = -c(region, X, X.x, X.y, X.x.1, X.y.1))

# is regional importance correlated within modality?
result_mri <- rcorr(as.matrix(coef_mri))
result_pet <- rcorr(as.matrix(coef_pet))

coef_mri$avg <- rowMeans(coef_mri)
coef_pet$avg <- rowMeans(coef_pet)

# is mean regional importance correlated across modalities?
cor.test(coef_mri$avg, coef_pet$avg)

# what are the most important regions
coef_pet$region <- coef_pet_0$region
coef_mri$region <- coef_mri_0$region
range(coef_pet$avg, na.rm = TRUE)
range(coef_mri$avg, na.rm = TRUE)
pet_upper <- mean(coef_pet$avg) + 2*sd(coef_pet$avg)
pet_lower <- mean(coef_pet$avg) - 2*sd(coef_pet$avg)
mri_upper <- mean(coef_mri$avg) + 2*sd(coef_mri$avg)
mri_lower <- mean(coef_mri$avg) - 2*sd(coef_mri$avg)
# range PET: [-.99, 1.04] --> important regions = <-.5 and >.5
coef_pet$region[(coef_pet$avg < pet_lower)]
coef_pet$region[(coef_pet$avg > pet_upper)]
# range MRI: [-.30, .22] --> important regions = <-.15 and >.15
coef_mri$region[(coef_mri$avg < mri_lower)]
coef_mri$region[(coef_mri$avg > mri_upper)]

# scatter mean signal
networks <- c("Vis", "SomMot", "TempPar", "DorsAttn", "SalVentAttn",
              "Cont", "Default", "Limbic", "-")
color <- c("darkorchid4", "darkturquoise", "chocolate4", "darkolivegreen",
           "darksalmon", "brown1", "darkred", "darkorange2", "black")

coef_mri$color <- NA
coef_mri$hemi <- NA
for (i in 1:nrow(coef_mri)){
  if (grepl('lh', coef_mri$region[i]) | (grepl('LH', coef_mri$region[i]))){
    coef_mri$hemi[i] <- 'LH'
  } else{coef_mri$hemi[i] <- 'RH'}
  for (j in 1:length(networks)){
    if (grepl(networks[j], coef_mri$region[i])){
      coef_mri$color[i] <- color[j]
    }
  }
}

coef_pet$color <- NA
coef_pet$hemi <- NA
for (i in 1:nrow(coef_pet)){
  if (grepl('lh', coef_pet$region[i]) | (grepl('LH', coef_pet$region[i]))){
    coef_pet$hemi[i] <- 'LH'
  } else{coef_pet$hemi[i] <- 'RH'}
  for (j in 1:length(networks)){
    if (grepl(networks[j], coef_pet$region[i])){
      coef_pet$color[i] <- color[j]
    }
  }
}

cor.test(coef_pet$avg[coef_pet$hemi=='LH'], coef_mri$avg[coef_mri$hemi=='LH'])
cor.test(coef_pet$avg[coef_pet$hemi=='RH'], coef_mri$avg[coef_mri$hemi=='RH'])

coefs <- merge(subset(coef_pet, select=c(region, avg, color, hemi)),
               subset(coef_mri, select=c(region, avg, color, hemi)),
               by="region")

new_scale <- function(new_aes) {
  structure(ggplot2::standardise_aes_names(new_aes), class = "new_aes")
}

g <- ggplot(coefs) +
  theme_classic() +
  geom_smooth(method='lm', aes(x = avg.x, y = avg.y, fill=hemi.x,
                               color=hemi.x)) +
  scale_fill_manual(values = c("LH" = "gray12",
                               "RH" = "azure4")) +
  scale_color_manual(values = c("LH" = "gray12",
                               "RH" = "azure4")) +
  new_scale_fill() +
  new_scale_color() +
  geom_point(aes(x = avg.x, y = avg.y, color=factor(color.x),
                 fill = factor(color.x), size=5), alpha=.8) +
  xlab(expression(paste(delta, " FDG-PET"))) +
  ylab(expression(paste(delta, " MRI"))) +
  theme(text = element_text(size=20)) +
  
  scale_color_manual(values = color,
                     labels = networks) +
  scale_fill_manual(values = color,
                    labels = networks,
                    guide = "none")
g
ggsave(filename = "weight_correlation.png",
       path = "2_BrainAge/PET_MRI_age_rev1/results/ADNI/CN/evaluation",
       width = 8, height = 5, device='tiff', dpi=300)
