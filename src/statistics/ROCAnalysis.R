#######################################
### PREDICTION OF PROGRESSION TO CI ###
#######################################
library(pROC)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(oddsratio)
library(MatchIt)
library(caret)
library(ggpattern)
library(patchwork)

# TODO: incomplete sets color change for MCI
# PLOTTING CONFIGURATION
rm(list=ls())
set.seed(0)
group <- "MCI"
pal.amy <- c(
  "neg" = "lightskyblue",
  "NA" = "white",
  "pos" = "tan1")
pal.apoe <- c(
  "1" = "burlywood2",
  "2" = "darkblue")
shape <- c(
  "CN" = 21,
  "MCI" = 22,
  "Dementia" = 23)
if (group == "CN"){
  size <- c(
    "CN" = 4,
    "MCI" = 6,
    "Dementia" = 6)
  alpha <- c(
    "CN"=0.4,
    "MCI"=0.9,
    "Dementia"=0.9)
} else{
  size <- c(
    "MCI" = 4,
    "Dementia" = 6)
  alpha <- c(
    "MCI"=0.4,
    "Dementia"=0.9)
}

# LOAD DATA
clin_data_bl <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_bl.csv",
                          na.strings = "")
diagnosis_24months <- read.csv2("2_BrainAge/PET_MRI_age/data/clin_data_24months2.csv",
                                na.strings = "")
diagnosis_earlier <- read.csv(paste(
  "2_BrainAge/PET_MRI_age/data/Diagnoses_upto2years_", group, ".csv", sep=""))
diagnosis_earlier <- subset(diagnosis_earlier, select = c(PTID, DX6, DX12, DX24))
colnames(clin_data_bl)[which(names(clin_data_bl) == "name")] <- "PTID"
clin_data_bl <- subset(clin_data_bl, select = -c(DX))
diagnosis_24months <- subset(diagnosis_24months, select = c(PTID, DX))

clin_data <- merge(clin_data_bl, diagnosis_24months, by = "PTID", all.x = TRUE)

model <- ifelse(group == "MCI", "_0", "")
pred_age_pet <- read.csv(paste("2_BrainAge/PET_MRI_age/results/", group, "/PET",
                           "-predicted_age_", group, model, ".csv", sep = ""))
pred_age_mri <- read.csv(paste("2_BrainAge/PET_MRI_age/results/", group, "/MRI",
                               "-predicted_age_", group, model, ".csv",
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
df <- merge(df, diagnosis_earlier, by='PTID', all.x = TRUE)

# create co-variates
df$APOE4c <- factor(ifelse(df$APOE4==0, 1, 2))
df$PTGENDER <- factor(ifelse(df$PTGENDER == "Female", 1, 0))
df$meanage <- rowMeans(df[,c(2,4)])
df$ABETA <- as.numeric(df$ABETA)
df$AMY.cat <- factor(
  ifelse(is.na(as.numeric(df$ABETA)), 0,
         ifelse(as.numeric(df$ABETA)>=1100, -1, 1)))
df$AMY.cat.c <- ifelse(df$AMY.cat == -1, "neg",
                       ifelse(df$AMY.cat == 0, "NA", "pos"))

# see if there is a correlation between PET and MRI BPAD
cor.test(df$PETBPAD, df$MRIBPAD)

# CONVERSION ANALYSIS
# if group is CN, consider conversion to MCI/AD
# if diagnosis of cognitive decline was given in previous years,
# and no diagnosis is available at year 2, use diagnosis of previous years
# do not use CN diagnoses from previous years if year 2 is NA
# (individuals may have converted by year 2, discard these individuals)
if (group == "CN") {
  df$DX_final <- ifelse(!is.na(df$DX), df$DX,
                        ifelse(df$DX12 == "MCI", "MCI",
                               ifelse(df$DX6 == "MCI", "MCI", NA)))
  df$DX.cat.n <- factor(ifelse(df$DX_final == "CN", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "CN", "X0_stable", "X1_decl"))
  df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
} else {
  # if group is MCI, remove diagnosis improvements ("CN")
  # and only consider conversion to Dementia  
  df$DX_final <- ifelse(!is.na(df$DX), df$DX,
                        ifelse(df$DX12 == "Dementia", "Dementia",
                               ifelse(df$DX6 == "Dementia", "Dementia", NA)))
  table(df$DX_final)
  df <- df[df$DX_final != "CN",]
  df$DX.cat.n <- factor(ifelse(df$DX_final == "MCI", 0, 1))
  df$DX.cat.c <- factor(ifelse(df$DX_final == "MCI", "X0_stable", "X1_decl"))
  df <- df[(!is.na(df$DX.cat.n)) & (!is.na(df$PTGENDER)),]
  
}

df$PETBPAD.cat <- ifelse(df$PETBPAD<0,-1,ifelse(df$PETBPAD==0,0,1))
df$MRIBPAD.cat <- ifelse(df$MRIBPAD<0,-1,ifelse(df$MRIBPAD==0,0,1))

table(df$DX.cat.c)

# whether to exclude incomplete cases
whole.set <- T
if (whole.set == F){
  df <- df[df$AMY.cat!=0,]
}

# propensity score matching with age of respective modality
p.match1 <- matchit(DX.cat.n ~ PTGENDER + meanage,
                     data = df, method = "optimal", distance = "glm")

summary(p.match1)

df.match1 <- match.data(p.match1)

unmatched <- df[!(df$PTID %in% df.match1$PTID) | df$DX.cat.c == "X1_decl",]
table(unmatched$DX.cat.c)
p.match2 <- matchit(DX.cat.n ~ PTGENDER + meanage,
                    data = unmatched, method = "optimal", distance = "glm")

df.match2 <- match.data(p.match2)
# see if there is a correlation between PET and MRI BPAD
cor.test(df.match1$PETBPAD, df.match1$MRIBPAD)
cor.test(df.match2$PETBPAD, df.match2$MRIBPAD)

# loop over two different matched sets
samples <- list(df.match1, df.match2)
cognitive.impairment <- function(x) {
  fitControl <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = 'final',
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  # use both MRI and PET BPAD as they are not strongly correlated in CN
  # additionally show models with only MRI or PET BPAD for MCI
  # due to strong correlation of BPAD
  # Ranganathan et al. (2017)
  traindata <- subset(x,
                      select=c(PETBPAD, MRIBPAD, AMY.cat, APOE4c, PTEDUCAT))
  trainclasses <- factor(x$DX.cat.c)
  nrow(traindata) == length(trainclasses)
  
  lrFit <- train(traindata, trainclasses,
                 method = "glm", family = "binomial",
                 trControl = fitControl)
}
set.seed(42)
models <- lapply(samples, cognitive.impairment)
s <- lapply(models, summary)
s

cognitive.impairment.sig <- function(x) {
  fitControl <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = 'final',
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  # use both MRI and PET BPAD as they are not strongly correlated in CN
  # additionally show models with only MRI or PET BPAD for MCI
  # due to strong correlation of BPAD
  # Ranganathan et al. (2017)
  traindata <- subset(x,
                      select=c(MRIBPAD, AMY.cat, APOE4c))
  trainclasses <- factor(x$DX.cat.c)
  nrow(traindata) == length(trainclasses)
  
  lrFit <- train(traindata, trainclasses,
                 method = "glm", family = "binomial",
                 trControl = fitControl)
  lrFit
}
set.seed(42)
models.sig <- lapply(list(df.match1), cognitive.impairment.sig)
s.sig <- lapply(models.sig, summary)
s.sig
models.sig[[1]]$results

# extract ground truth (gt), predictions (pred) and probabilities (prob)
# from model
sample = 1
prediction.df <- data.frame(rowIndex = models.sig[[1]]$pred$rowIndex,
                            gt = models.sig[[1]]$pred$obs,
                            pred = models.sig[[1]]$pred$pred,
                            prob = models.sig[[1]]$pred$X1_decl)
# add relevant variables to prediction df
for (j in 1:nrow(prediction.df)){
  prediction.df$PTID[j] <- samples[[sample]]$PTID[prediction.df$rowIndex[j]]
  prediction.df$DX[j] <- samples[[sample]]$DX_final[prediction.df$rowIndex[j]]
  prediction.df$PETBPAD[j] <- samples[[sample]]$PETBPAD[prediction.df$rowIndex[j]]
  prediction.df$MRIBPAD[j] <- samples[[sample]]$MRIBPAD[prediction.df$rowIndex[j]]
  prediction.df$PETAGE[j] <- samples[[sample]]$PETAGE[prediction.df$rowIndex[j]]
  prediction.df$MRIAGE[j] <- samples[[sample]]$MRIAGE[prediction.df$rowIndex[j]]
  prediction.df$APOE[j] <- samples[[sample]]$APOE4[prediction.df$rowIndex[j]]
  prediction.df$APOEc[j] <- samples[[sample]]$APOE4c[prediction.df$rowIndex[j]]
  prediction.df$AMY.cat.c[j] <- samples[[sample]]$AMY.cat.c[prediction.df$rowIndex[j]]
}

prediction.df$APOEc <- as.character(prediction.df$APOEc)
prediction.df$AMY.cat <- as.character(prediction.df$AMY.cat)

modality <- "MRI"
get.odds <- function(sample){
  # save significance
  coeff <- data.frame(s[[sample]]['coefficients'])
  write.csv(coeff, 
            paste("2_BrainAge/PET_MRI_age/results/conv/", 
                  group, "_CONV_",
                  as.character(whole.set), "model_", sample, "_NEW.csv", sep = ""))
  # plot and save odds ratios
  OR <- or_glm(data = samples[[sample]], model = models[[sample]]$finalModel, 
               incr = list(PETBPAD = 1, MRIBPAD = 1, AMY.cat = 1, APOE4 = 1,
                           PTEDUCAT = 1))
  print(OR)
  g <- ggplot(OR, aes(x = oddsratio, y = predictor)) + 
    windows(width=6, height=6) +
    geom_vline(aes(xintercept = 0), size = .25, linetype = "dashed") + 
    geom_errorbarh(aes(xmax = `ci_high (97.5)`, 
                       xmin = `ci_low (2.5)`), size = 1, height = 0.1,
                   color = "gray50") +
    geom_point(size = 6, color = "orange") +
    theme_bw()+
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face="bold"),
          axis.title = element_text(face="bold")) +
    ylab("") +
    xlab("Odds ratio")
  ggsave(filename = paste(group, "_", modality, "_ODDS_CONV_compl_", sample, 
                          as.character(whole.set), "_NEW.png", sep=""),
         path = "2_BrainAge/PET_MRI_age/results/conv/",
         width = 10, height = 10, device='tiff', dpi=300)
  write.csv(OR, 
            paste("2_BrainAge/PET_MRI_age/results/conv/", 
                  group, "_oddsratios_CONV_",
                  as.character(whole.set),"_model_",
                  sample, "_NEW.csv", sep = ""), row.names = F)
}
get.cutoff <- function(sample){
  # PLOTTING
  prediction.df <- prediction.df[!(prediction.df$AMY.cat.c=="NA"),]
  attach(prediction.df)
  # plot logistic regression probabilities with BPAD and most important 
  # non-BPAD feature (APOE for both MCI, and one CN group(s))
  detach(prediction.df)
  attach(prediction.df)
  if ((sample == 2) & (group == "CN")){
    pal <- pal.amy
    var_ <- AMY.cat.c
  } else if ((sample == 1) & (group == "CN")){
    pal <- "gray"
    var_ <- "gray"
  } else{
    pal <- pal.apoe
    var_ <- APOEc
  }
  #alpha <- c(
  #  "neg"=0.5,
  #  "pos"=1)
  g <- ggplot(prediction.df, aes(x = MRIBPAD, y = prob, pattern = AMY.cat.c)) +
    theme_classic() +
    geom_smooth(method = "glm", 
                method.args = list(family = "binomial"), 
                aes(color = var_), alpha = 0.2, se = F, linetype = "dashed") +
    geom_smooth(method = "glm", 
                method.args = list(family = "binomial"), 
                color = "gray", alpha = 0.2, se = F) +
    geom_point(aes(shape=DX, size=DX, alpha=DX, color = var_,
                   fill = var_)) +
    geom_hline(yintercept = 0.5, linetype = "dotted", col = "gray", size = 2) +
    annotate("text", x = min(MRIBPAD)+2, y = 0.5, col = "gray",
             label = "Cut-off", vjust = -0.5, size = 5) +
    theme(
      axis.title = element_text(size = 11, face = "bold"),
      plot.title = element_text(face = "bold"),
      legend.title = element_text(size = 15, face = "bold")) +
    scale_alpha_manual(guide="none",
                       values = alpha) +
    scale_shape_manual(values = shape,
                       limits  = names(shape)) +
    scale_size_manual(guide="none",
                      values=size) +
    scale_color_manual(values = pal,
                       limits = names(pal)) +
    scale_fill_manual(values = pal,
                      limits = names(pal)) +
    scale_pattern_manual(values = c(neg = "none",
                                    pos = "stripe")) +
    ylab("Probability of Disease Progression") +
    xlab("BPAD") +
    theme(text = element_text(size=20))
  dens1 <- ggplot(prediction.df, aes(x = MRIBPAD, pattern = gt,
                                     fill = var_, alpha = 0.5)) + 
    geom_density_pattern(position = position_dodge(preserve = "single"),
                     # pattern_fill = pal,
                     pattern_angle = 45,
                     pattern_density = 0.1,
                     pattern_spacing = 0.025,
                     pattern_key_scale_factor = 0.6) +
    scale_pattern_manual(values = c(X0_stable = "none",
                                    X1_decl = "stripe")) +
    scale_fill_manual(values = pal,
                      limits = names(pal)) +
    # scale_color_manual()
    theme_void()
    library(patchwork)
    
  dens1 + plot_spacer() + g + 
      plot_layout(
        ncol = 2, 
        nrow = 2, 
        widths = c(6, 1),
        heights = c(1, 4)
      ) 
  # get intersection of line with 50% probability
  cutoff <- ggplot_build(g)$data[[1]]
  print(cutoff[(0.48<cutoff$y) & (0.52>cutoff$y),])
  
  ggsave(filename = paste(group, "_", modality, "_CONV_compl", sample, 
                          as.character(whole.set), "_NEW.png", sep=""),
         path = "2_BrainAge/PET_MRI_age/results/conv/",
         width = 10, height = 10, device='tiff', dpi=300)
  detach(prediction.df)
}
get.odds(sample)
get.cutoff(sample)

# CN LOOK AT SENS IN AMY 
amy_neg2 <- df.match2[df.match2$AMY.cat.c == "neg",]
amy_neg2$BPAD_older <- ifelse(amy_neg2$PETBPAD>2.4,"yes", "no")
table(amy_neg2$BPAD_older, amy_neg2$DX.cat.c)
amy_pos2 <- df.match2[df.match2$AMY.cat.c == "pos",]
amy_pos2$BPAD_older <- ifelse(amy_pos2$PETBPAD>-2.6,"yes", "no")
table(amy_pos2$BPAD_older, amy_pos2$DX.cat.c)
# MCI by APOE
apoe_neg1 <- df.match1[df.match1$APOE4c == 1,]
apoe_neg1$BPAD_older <- ifelse(apoe_neg1$MRIBPAD>3.6,"yes", "no")
table(apoe_neg1$BPAD_older, apoe_neg1$DX.cat.c)
apoe_pos1 <- df.match1[df.match1$APOE4c == 2,]
apoe_pos1$BPAD_older <- ifelse(apoe_pos1$MRIBPAD>.8,"yes", "no")
table(apoe_pos1$BPAD_older, apoe_pos1$DX.cat.c)

apoe_neg2 <- df.match2[df.match2$APOE4c == 1,]
apoe_neg2$BPAD_older <- ifelse(apoe_neg2$MRIBPAD>5.2,"yes", "no")
table(apoe_neg2$BPAD_older, apoe_neg2$DX.cat.c)
apoe_pos2 <- df.match2[df.match2$APOE4c == 2,]
apoe_pos2$BPAD_older <- ifelse(apoe_pos2$MRIBPAD>.3,"yes", "no")
table(apoe_pos2$BPAD_older, apoe_pos2$DX.cat.c)

# MCI
df.match1$BPAD_older <- ifelse(df.match1$MRIBPAD>2.3,"yes", "no")
df.match2$BPAD_older <- ifelse(df.match2$MRIBPAD>2.7, "yes", "no")
table(df.match1$BPAD_older, df.match1$DX.cat.c)
table(df.match2$BPAD_older, df.match2$DX.cat.c)
table(df.match1$AMY.cat, df.match1$DX.cat.c)
table(df.match2$AMY.cat, df.match2$DX.cat.c)
table(df.match1$APOE4c, df.match1$DX.cat.c)
table(df.match2$APOE4c, df.match2$DX.cat.c)

 

### REFERENCES ###
# Ranganathan, P., Pramesh, C. S., & Aggarwal, R. (2017). 
# Common pitfalls in statistical analysis: Logistic regression. 
# Perspectives in clinical research, 8(3), 148-151.
# https://doi.org/10.4103/picr.PICR_87_17