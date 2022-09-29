### CALCULATE CORRELATION MATRIX OF FEATURE IMPORTANCE
library(corrplot)
library(ggplot2)
library(viridis)
rm(list=ls())
modality = "MRI"
m0 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                 "weighted_importance_", modality, "_svm_0.csv", sep=""))
m1 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_1.csv", sep=""))
m2 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_MRI_svm_2.csv", sep=""))
m3 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_MRI_svm_3.csv", sep=""))
m4 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_4.csv", sep=""))

modality = "PET"
m0.pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_0.csv", sep=""))
m1.pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_1.csv", sep=""))
m4.pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/CN/evaluation/",
                     "weighted_importance_", modality, "_svm_4.csv", sep=""))

# absolute values for undirected feature importance
df.mri <- data.frame(m0$perm_importance, m1$perm_importance,
                     m2$perm_importance, m3$perm_importance, 
                     m4$perm_importance)
df.pet <- data.frame(m0.pet$perm_importance, m1.pet$perm_importance,
                     m4.pet$perm_importance)
# df$means <- rowMeans(df)
#df.mri$region <- m0$region
#df.pet$region <- m0.pet$region
# most important regions
# q <- quantile(df$means, probs = seq(0,1,0.9))
# df$region[df$means>quantile(df$means)['90%']]


# make mean
cor.test(m0$perm_importance, m0.pet$perm_importance)
cor.test(m1$perm_importance, m1.pet$perm_importance)
cor.test(m4$perm_importance, m4.pet$perm_importance)

# cor(m1$perm_importance[m1$X>200], m1_mri$perm_importance[m1_mri$X>200])
cor.mat.mri <- cor(df.mri)
min(cor.mat.mri)
cor.mat.pet <- cor(df.pet)
min(cor.mat.pet)
corrplot(cor.mat.mri, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 45)

df.mri$mean <- rowMeans(df.mri)
df.pet$mean <- rowMeans(df.pet)
cor.test(df.mri$mean, df.pet$mean)


### CALCULATE CORRELATION OF BRAIN AGE ACROSS MODELS IN MCI
modality <- "PET"
m0 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/MCI/", modality,
                     "-predicted_age_MCI_0.csv", sep=""))
m1 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/MCI/", modality,
                     "-predicted_age_MCI_1.csv", sep=""))
m2 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/MCI/", modality,
                     "-predicted_age_MCI_2.csv", sep=""))
m3 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/MCI/", modality,
                     "-predicted_age_MCI_3.csv", sep=""))
m4 <- read.csv(paste("2_BrainAge/PET_MRI_age/results/MCI/", modality,
                     "-predicted_age_MCI_4.csv", sep=""))
m0$BPAD <- m0$Prediction - m0$Age
m1$BPAD <- m1$Prediction - m1$Age
m2$BPAD <- m2$Prediction - m2$Age
m3$BPAD <- m3$Prediction - m3$Age
m4$BPAD <- m4$Prediction - m4$Age
df <- data.frame(m0$BPAD, m1$BPAD,
                 m2$BPAD, m3$BPAD, 
                 m4$BPAD)
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
