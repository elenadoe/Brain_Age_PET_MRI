#######################################
###     Prediction of DX change     ###
#######################################
library(ggplot2)
library(caret)
library(MatchIt)
library(oddsratio)
library(ggpattern)
library(patchwork)

rm(list=ls())
group <- 'CN'
database <- 'ADNI'
shape <- c(
  "X0_stable" = 21,
  "X1_decl" = 23)
size <- c(
    "X0_stable" = 4,
    "X1_decl" = 6)
alpha <- c(
    "X0_stable"=0.4,
    "X1_decl"=0.9)

set.seed(0)

df <- read.csv(
  paste("2_BrainAge/PET_MRI_age0/data/ADNI/PsychPath/",
        sprintf("merged_for_dx_prediction_%s.csv", group), sep = ""))

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

# unmatched <- df[!(df$PTID %in% df.match1$PTID) | df$DX.cat.c == "X1_decl",]
# table(unmatched$DX.cat.c)
#p.match2 <- matchit(DX.cat.n ~ PTGENDER + meanage,
#                    data = unmatched, method = "optimal", distance = "glm")
#
# <- match.data(p.match2)

# loop over two different matched sets
#samples <- list(df.match1, df.match2)

cognitive.impairment <- function(x) {
  fitControl <- trainControl(
    method = "cv", number = 10, savePredictions = 'final',
    classProbs = TRUE, summaryFunction = twoClassSummary
  )
  
  # use both MRI and PET BPAD as they are not strongly correlated in CN
  # additionally show models with only MRI or PET BPAD for MCI
  # due to strong correlation of BPAD
  # Ranganathan et al. (2017)
  traindata <- subset(x,
                      select=c(PET.BAG, MRI.BAG, ABETA.cat, APOE4, PTEDUCAT))
  trainclasses <- factor(x$DX.cat.c)
  nrow(traindata) == length(trainclasses)
  
  lrFit <- train(traindata, trainclasses,
                 method = "glm", family = "binomial",
                 trControl = fitControl)
}
set.seed(0)
model <- cognitive.impairment(df.match1)
s <- summary(model)
s

prediction.df <- data.frame(rowIndex = model$pred$rowIndex,
                            gt = model$pred$obs,
                            pred = model$pred$pred,
                            prob = model$pred$X1_decl)
# add relevant variables to prediction df
for (j in 1:nrow(prediction.df)){
  prediction.df$PTID[j] <- df.match1$PTID[prediction.df$rowIndex[j]]
  prediction.df$DX[j] <- df.match1$DX_final[prediction.df$rowIndex[j]]
  prediction.df$PET.BAG[j] <- df.match1$PET.BAG[prediction.df$rowIndex[j]]
  prediction.df$MRI.BAG[j] <- df.match1$MRI.BAG[prediction.df$rowIndex[j]]
  prediction.df$PETAGE[j] <- df.match1$PETAGE[prediction.df$rowIndex[j]]
  prediction.df$MRIAGE[j] <- df.match1$MRIAGE[prediction.df$rowIndex[j]]
  prediction.df$APOE[j] <- df.match1$APOE4[prediction.df$rowIndex[j]]
  prediction.df$AMY.cat[j] <- df.match1$ABETA.cat[prediction.df$rowIndex[j]]
}

get.odds <- function(x){
  # save significance
  coeff <- data.frame(s$coefficients)
  write.csv(
    coeff, 
    sprintf(
      "2_BrainAge/PET_MRI_age0/results/%s/%s/2_DX_change/%s_significancevalues.csv",
      database, group, group))
  # save odds ratios
  OR <- or_glm(data = x, model = model$finalModel, 
               incr = list(PETBAG = 1, MRIBAG = 1, AMY.cat = 1, APOE4 = 1,
                           PTEDUCAT = 1))
  print(OR)
  write.csv(
    OR, 
    sprintf("2_BrainAge/PET_MRI_age0/results/%s/%s/2_DX_change/%s_oddsratios.csv", 
            database, group, group), row.names = F)
}
get.odds(model)

attach(prediction.df)
winner <- PET.BAG
winner.name <- ifelse(winner == MRI.BAG, 'MRI', 'PET')[1]
color <- ifelse(winner == MRI.BAG, 'cyan4', 'coral')[1]


g <- ggplot(prediction.df, aes(x = winner, y = prob)) +
  theme_classic() +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"), 
              aes(color = color, fill = color), alpha = 0.2, se = T, linetype = "dashed") +
  # scatter individual data points
  geom_point(aes(shape = gt, size = gt, alpha = gt, color = color,
                 fill = color)) +
  # 50% threshold
  geom_hline(yintercept = 0.5, linetype = "dotted", col = "gray", size = 2) +
  annotate("text", x = min(winner)+2, y = 0.5, col = "gray",
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
print(cutoff[(0.48<cutoff$y) & (0.52>cutoff$y),])

ggsave(filename = sprintf(
  "%s_DX_change_prediction_%s_%s.png", group, database, winner.name),
  path = sprintf("2_BrainAge/PET_MRI_age0/results/%s/%s/2_DX_change/",
                 database, group),
  width = 10, height = 10, device='tiff', dpi=300)
detach(prediction.df)
