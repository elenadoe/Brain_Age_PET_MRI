### CALCULATE CORRELATION MATRIX OF FEATURE IMPORTANCE
library(corrplot)
library(ggplot2)
library(viridis)
rm(list=ls())
modality = "PET"
m0 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                 "weighted_importance_", modality, "_svm_0.csv", sep=""))
m1 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_1.csv", sep=""))
#m2 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
#                     "weighted_importance_MRI_svm_2.csv", sep=""))
#m3 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
#                     "weighted_importance_MRI_svm_3.csv", sep=""))
m4 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_4.csv", sep=""))
df <- data.frame(m0$perm_importance, m1$perm_importance,
                   #m2$perm_importance, m3$perm_importance, 
                 m4$perm_importance)
cor.mat <- cor(df)
min(cor.mat)
corrplot(cor.mat, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 45)

### CALCULATE CORRELATION OF BRAIN AGE ACROSS MODALITIES
p.mri <- read.csv("2_BrainAge/PET_MRI_age/results/CN/MRI-predicted_age_CN.csv")
p.mri$BPAD <- p.mri$Prediction - p.mri$Age
p.pet <- read.csv("2_BrainAge/PET_MRI_age/results/CN/PET-predicted_age_CN.csv")
p.pet$BPAD <- p.pet$Prediction - p.pet$Age
df2 <- data.frame(p.mri$PTID, p.mri$Age, p.mri$BPAD, p.pet$BPAD)
colnames(df2) <- c("PTID", "Age", "MRI.BPAD", "PET.BPAD")

dem <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_bl.csv",
                  na.strings = "")

df2 <- merge(df2, dem, by="PTID", all.x = TRUE)

cor(p.mri$Prediction, p.pet$Prediction)  
cor.test(p.mri$BPAD, p.pet$BPAD)
plot(p.mri$BPAD, p.pet$BPAD)

df2$AGECAT <- ifelse(df2$Age<74,"young old", ifelse(df2$Age<84,"middle old",
                                                    "oldest old"))
df2$APOE4 <- as.factor(df2$APOE4)

library(patchwork)
p <- ggplot(df2, aes(x=MRI.BPAD, y=PET.BPAD)) +
  theme_bw() +
  #geom_point() +
  geom_point(aes(color=APOE4)) +
  #scale_color_viridis(option="D") +
  #geom_point(aes(color=PTGENDER)) +
  scale_color_viridis(option="D", discrete=TRUE) +
  theme(
    axis.title = element_text(size = 11, face = "bold"),
    plot.title = element_text(face = "bold"),
    legend.title = element_text(size = 15, face = "bold")) +
  ylab("PET-BPAD") +
  xlab("MRI-BPAD") +
  ggtitle("PET- by MRI-BPAD")

dens1 <- ggplot(df2, aes(x = MRI.BPAD)) + 
  #geom_histogram() +
  geom_histogram(aes(color = APOE4, fill = APOE4), alpha=0.65) + 
  scale_color_viridis(option="D", discrete=TRUE) +
  scale_fill_viridis(option="D", discrete=TRUE) +
  theme_void()

dens2 <- ggplot(df2, aes(x = PET.BPAD)) + 
  #geom_histogram() +
  geom_histogram(aes(color = APOE4, fill=APOE4), alpha=0.65) + 
  scale_color_viridis(option="D", discrete=TRUE) +
  scale_fill_viridis(option="D", discrete=TRUE) +
  theme_void() + 
  coord_flip()

dens1 + plot_spacer() + p + dens2 + 
  plot_layout(
    ncol = 2, 
    nrow = 2, 
    widths = c(4, 1),
    heights = c(1, 4)
  )


df2$MRIBPADCAT <- ifelse(df2$MRI.BPAD>0,1, ifelse(df2$MRI.BPAD==0,0,-1))
df2$PETBPADCAT <- ifelse(df2$PET.BPAD>0,1, ifelse(df2$PET.BPAD==0,0,-1))
