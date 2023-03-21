#######################################
###     Prediction of DX change     ###
#######################################
library(ggplot2)
library(caret)
library(MatchIt)
library(oddsratio)
library(ggpattern)
library(patchwork)
library(MASS)
library(predtools)

rm(list=ls())

group <- 'CN'
atlas <- 'AAL1_cropped'
database <- 'ADNI'
matching <- T
whole.set <- T
shape <- c(
  "X0_stable" = 21,
  "X1_decl" = 23)
size <- c(
    "X0_stable" = 4,
    "X1_decl" = 6)
alpha <- c(
    "X0_stable"=0.4,
    "X1_decl"=0.9)

if (group != 'all' & group != 'CU'){
  df <- read.csv(sprintf(
    '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/%s/merged_for_dx_prediction_%s_%s.csv',
    group, atlas, group))
} else{
  df <- read.csv(sprintf(
    '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/merged_for_dx_prediction_%s_%s.csv',
    atlas, group))
  df$DX.n <- ifelse(df$DX.bl == 'CN', 1, ifelse(
    df$DX.bl == 'SMC', 2, 3))
}
df <- df[!is.na(df$DX.cat.c),]
row.names(df) <- NULL
df$PTAUAB.cat <- as.factor(df$PTAUAB42.cutoff)
df$AB.cat <- as.factor(df$ABETA42.cutoff)

df$PTGENDER <- as.factor(df$PTGENDER)
# df$PTEDUCAT <- as.factor(ifelse(df$PTEDUCAT<16,0,1))
df$APOE4 <- ifelse(df$APOE4>0,1,0)
df$APOE4 <- as.factor(df$APOE4)
df$DX.bl <- as.factor(ifelse(df$DX.bl == "CN", 1, ifelse(
  df$DX.bl == "SMC", 2, 3
)))

if (whole.set == F){
  df <- df[df$PTAUAB.cat!=0,]
}

if (matching){
  whole.set <- T
  
  
  # propensity score matching with age of respective modality
  set.seed(0)
  p.match1 <- matchit(DX.cat.n ~ PTGENDER + meanage,
                      data = df, method = "optimal", distance = "glm")
  
  summary(p.match1)
  
  df <- match.data(p.match1)
  
  # unmatched <- df[!(df$PTID %in% df.match1$PTID) | df$DX.cat.c == "X1_decl",]
  # table(unmatched$DX.cat.c)
  #p.match2 <- matchit(DX.cat.n ~ PTGENDER + meanage,
  #                    data = unmatched, method = "optimal", distance = "glm")
  #
  # <- match.data(p.match2)
  
  # loop over two different matched sets
  #samples <- list(df.match1, df.match2)
}

cognitive.impairment <- function(x) {
  fitControl <- trainControl(
    method = "cv", number = 10, savePredictions = 'final',
    classProbs = TRUE, summaryFunction = twoClassSummary
  )
  
  # use both RI and PET BPAD as they are not strongly correlated in CN
  # additionally show models with only MRI or PET BPAD for MCI
  # due to strong correlation of BPAD
  # Ranganathan et al. (2017)
  traindata <- subset(x,
                      select=c(PET.BAG, MRI.BAG, AB.cat,
                               APOE4, PTEDUCAT, meanage, PTGENDER))
  trainclasses <- factor(x$DX.cat.c)
  nrow(traindata) == length(trainclasses)
  
  lrFit <- train(traindata, trainclasses,
                 method = "glm", family = "binomial",
                 trControl = fitControl)
}
set.seed(0)
model <- cognitive.impairment(df)
s <- summary(model)
s
val_data <- model$pred
val_data$obs <- ifelse(val_data$obs == 'X0_stable', 0, 1)
calibration_plot(data = val_data, obs = 'obs', pred = 'X1_decl',
                 title = "Calibration plot for validation data",
                 x_lim=c(0,max(val_data$X1_decl)), y_lim=c(0,max(val_data$X1_decl)))

em <- function(y, posteriors_zero, priors_zero, epsilon=1e-6, positive_class=1) {
  s <- 0
  priors_s <- priors_zero
  posteriors_s <- posteriors_zero
  val <- 2 * epsilon
  history <- list()
  acc <- mean((y == positive_class) == (posteriors_zero[, positive_class] > 0.5))
  rec <- sum((y == positive_class) & (posteriors_zero[, positive_class] > 0.5)) / sum(y == positive_class)
  prec <- sum((y == positive_class) & (posteriors_zero[, positive_class] > 0.5)) / sum(posteriors_zero[, positive_class] > 0.5)
  history[[1]] <- list(s=s, priors_s=as.list(priors_s), val=1, acc=acc, prec=prec, rec=rec)
  
  while (val >= epsilon) {
    # E step
    ratios <- priors_s / priors_zero
    denominators <- apply(ratios * posteriors_zero, 1, sum)
    for (c in seq_along(priors_zero)) {
      posteriors_s[, c] <- ratios[c] * posteriors_zero[, c] / denominators
    }
    acc <- mean((y == positive_class) == (posteriors_s[, positive_class] > 0.5))
    rec <- sum((y == positive_class) & (posteriors_s[, positive_class] > 0.5)) / sum(y == positive_class)
    prec <- sum((y == positive_class) & (posteriors_s[, positive_class] > 0.5)) / sum(posteriors_s[, positive_class] > 0.5)
    
    # M step
    priors_s_minus_one <- priors_s
    posteriors_sum <- rowSums(posteriors_s)
    priors_s <- posteriors_sum / sum(posteriors_sum)
    
    # check for stop
    val <- sum(abs(priors_s - priors_s_minus_one))
    s <- s + 1
    history[[s+1]] <- list(s=s, priors_s=as.list(priors_s), val=val, acc=acc, prec=prec, rec=rec)
  }
  return(list(posteriors_s=posteriors_s, priors_s=priors_s, history=history))
}
priors_zero <- ifelse(group == 'CN' | group == 'SMC',
                      rep(.06, each=nrow(val_data)),
                      rep(.2, each=nrow(val_data)))
priors_zero <- 0.5
priors_zero <- table(val_data$obs)[2]/(table(val_data$obs)[1]+table(val_data$obs)[2])
e.max <- em(val_data$obs, data.frame(val_data$X0_stable, val_data$X1_decl), priors_zero)


# save odds ratios of full model
OR <- or_glm(data = df, model = model$finalModel, 
             incr = list(PET.BAG = 1, MRI.BAG = 1, PTEDUCAT = 1,
                         meanage = 1))
OR

prediction.df <- data.frame(rowIndex = model$pred$rowIndex,
                            gt = model$pred$obs,
                            pred = model$pred$pred,
                            prob = model$pred$X1_decl)
# add relevant variables to prediction df
for (j in 1:nrow(prediction.df)){
  prediction.df$PTID[j] <- df$PTID[prediction.df$rowIndex[j]]
  prediction.df$DX[j] <- df$DX_final[prediction.df$rowIndex[j]]
  prediction.df$PET.BAG[j] <- df$PET.BAG[prediction.df$rowIndex[j]]
  prediction.df$MRI.BAG[j] <- df$MRI.BAG[prediction.df$rowIndex[j]]
  prediction.df$PETAGE[j] <- df$PETAGE[prediction.df$rowIndex[j]]
  prediction.df$MRIAGE[j] <- df$MRIAGE[prediction.df$rowIndex[j]]
  prediction.df$APOE[j] <- df$APOE4[prediction.df$rowIndex[j]]
  prediction.df$AMY.cat[j] <- df$ABETA.cat[prediction.df$rowIndex[j]]
}
prediction.df$gt.num <- ifelse(prediction.df$gt == "X0_stable", 0, 1)

attach(prediction.df)
winner <- PET.BAG
winner.name <- ifelse(winner == MRI.BAG, 'MRI', 'PET')[1]
color <- ifelse(winner == MRI.BAG, 'midnightblue', 'darkred')[1]
# apoe_pal <- c("1" = 'coral3', "2" = 'chocolate')

g <- ggplot(prediction.df) +
  theme_classic() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"), 
              aes(x = winner, y = gt.num, color = color, fill = color), se = T, linetype = "dashed") +
  # scatter individual data points
  geom_point(aes(x = winner, y = prob, shape = gt, size = gt, alpha = gt, color = color,
                 fill = color)) +
  # 50% threshold
  geom_hline(yintercept = 0.5, linetype = "dotted", col = "black", size = 2) +
  annotate("text", x = min(winner)+1, y = 0.5, col = "black",
           label = "Cut-off", vjust = -0.5, size = 5) +

  # configure text size
  theme(
    axis.title = element_text(size = 11, face = "bold"),
    plot.title = element_text(face = "bold"),
    legend.title = element_text(size = 15, face = "bold")) +

  # configure color and shape
  scale_alpha_manual(guide = "none",
                     values = alpha) +
  scale_shape_manual(values = shape,
                     limits  = names(shape)) +
  scale_size_manual(guide = "none",
                    values = size) +
  scale_color_manual(guide = "none",
                     values = color) +
  scale_fill_manual(guide = "none",
                    values = color) +

  # configure axes
  ylab("Probability of DX change") +
  xlab("BAG") +
  theme(text = element_text(size=20))
  
# configure density plot
dens1 <- ggplot(prediction.df, aes(x = winner, pattern = gt,
                                   fill = color, alpha = 0.5)) + 
  geom_density_pattern(position = position_dodge(preserve = "single"),
                       pattern_angle = 45,
                       pattern_density = 0.1,
                       pattern_spacing = 0.025,
                       pattern_key_scale_factor = 0.6) +
  scale_pattern_manual(values = c(X0_stable = "none",
                                  X1_decl = "stripe")) +
  scale_fill_manual(values = color,
                    guide = "none") +
  theme_void()

# add plots together
dens1 + plot_spacer() + g + 
  plot_layout(
    ncol = 2, 
    nrow = 2, 
    widths = c(6, 1),
    heights = c(1, 4)
  ) 

# get intersection of line with ~50% probability
cutoff <- ggplot_build(g)$data[[1]]

print(cutoff[(0.48<cutoff$y) & (0.52>cutoff$y),]) # CN: 0.7829160 MCI: 2.230878
#print(prediction.df[(0.48<prediction.df$prob) & (0.52>prediction.df$prob),])
my_glm <- glm(gt~winner, data=prediction.df, family=binomial())
cutoff <- dose.p(my_glm, p=c(0.5)) # CN: 0.6670282 MCI: 2.130332 
cutoff[1]
if (group == 'CN'){
  # cutoff_final <- 0.8522485
  cutoff_final <- cutoff[1]
  df$threshold <- ifelse(df$PET.BAG > cutoff_final, "Y1_decl",
                                "Y0_stable")
  df.match1$threshold <- ifelse(df.match1$PET.BAG > cutoff_final, "Y1_decl",
                         "Y0_stable")
} else{
  #cutoff_final = 1.555705 
  cutoff_final <- cutoff[1]
  df$threshold <- ifelse(df$MRI.BAG > cutoff_final, 'Y1_decl',
                                'Y0_stable')
  df.match1$threshold <- ifelse(df.match1$MRI.BAG > cutoff_final, "Y1_decl",
                                "Y0_stable")
}

table(df$threshold, df$DX.cat.c)
table(df.match1$threshold, df.match1$DX.cat.c)

ggsave(filename = sprintf(
  "%s_DX_change_prediction_%s_%s.png", group, database, winner.name),
  path = sprintf("2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/2_DX_change",
                 database, group),
  width = 10, height = 10, device='tiff', dpi=300)
detach(prediction.df)
