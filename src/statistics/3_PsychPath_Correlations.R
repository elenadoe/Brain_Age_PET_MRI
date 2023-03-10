#######################################
###      CORRELATION WITH CI/NP     ###
#######################################
library(ggplot2)
library(ppcor)
library(car)

# TODO: add Hippocampus & Precuneus
rm(list=ls)
group <- 'SMC'
database <- 'ADNI'
atlas <- 'AAL1_cropped'
# color <- ifelse(modality == 'MRI', 'midnightblue', 'darkred')
psychpath <- 'PATH'
long_psychpath <- ifelse(
  psychpath == 'PATH', '1_pathology', '1_cognitive performance')

if (group != 'all' & group != 'CU'){
  pred_table <- read.csv(sprintf(
    '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/%s/merged_for_dx_prediction_%s_%s.csv',
    group, atlas, group))
  covars <- c('PTEDUCAT', 'APOE4', 'meanage', 'PTGENDER')
} else{
  pred_table <- read.csv(sprintf(
    '2_BrainAge/Brain_Age_PET_MRI/results/ADNI/merged_for_dx_prediction_%s_%s.csv',
    atlas, group))
  pred_table$DX.n <- ifelse(pred_table$DX.bl == 'CN', 1, ifelse(
    pred_table$DX.bl == 'SMC', 2, 3))
  covars <- c('PTEDUCAT', 'APOE4', 'meanage', 'PTGENDER', 'DX.n')
}

pred_table$AT <- factor(ifelse(pred_table$PTAU.ABETA42_0.023_cutoff==1, 1,
                               ifelse(is.nan(pred_table$PTAU.ABETA42_0.023_cutoff), NA, 0)))

# compare AT status
at <- function(df, bag){
  df <- df[!is.na(df$AT),]
  # df[, "AT"] <- factor(df[, "AT"], levels=c("A-T-", "A+T-", "A-T+", "A+T+"))
  colors <- c("midnightblue", "darkred")
  for (j in range(1,length(bag))){
    current_bag <- bag[j]
    print(current_bag)
    g <- ggplot(df) +
      geom_boxplot(aes(x=df[, "AT"], y=df[,current_bag]), fill=colors[j],
                   alpha=0.7, outlier.color = "black") +
      # geom_dotplot(aes(x=df[, "AT"], y=df[,current_bag]),
      #              binaxis="y", stackdir = "center", dotsize = 0.4) +
      xlab("\nA/T status") +
      ylab(paste(current_bag, " [years]")) +
      theme_classic() +
      scale_color_discrete() +
      theme(text = element_text(size=20))
    g
    ggsave(filename = paste(current_bag, '_PTAUAB_', atlas, ".png", sep=""),
           path = sprintf(
             "2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/1_pathology/",
             database, group),
           width = 10, height = 10, device='tiff', dpi=300)
    # Normal distribution of residuals
    
    # Homogeneity of variance
    # db <- describeBy(df[,current_bag], df$AT)
    # print(db)

    aov <- lm(df[,current_bag]~df$PTEDUCAT+df$APOE4+df$meanage+df$PTGENDER+df$AT)
    aov_type3 <- Anova(aov, type="III")
    print(aov_type3)

    posthoc <- pairwise.t.test(df[,current_bag], df$AT, p.adjust.method = "none")
    print(posthoc)
  }
}

bag <- c("MRI.BAG", "PET.BAG")
at(pred_table, bag)


if (psychpath == 'PATH'){
    vars_ <- c('SUMMARYSUVR_WHOLECEREBNORM', 'ABETA42_recalculated',
               'TAU', 'PTAU', 'PTAU.ABETA42')
} else{
  vars_ <- c('ADNI_MEM', 'ADNI_EF')
}

# FIND PARTIAL CORRELATIONS
get_cors <- function(df, vars_, covars, bag, amy=''){
  for (i in 1:length(vars_)){
  # exclude missing values in var of interest
  missing <- is.na(df[, vars_[i]]) | is.na(df$APOE4) # pcor can't handle missing values
  df2 <- df[!missing,]
  
  # assess normality
  sw <- shapiro.test(df2[, vars_[i]])
  if (sw$p.value < 0.05){
    print(paste(vars_[i], sprintf("(n = %i)", nrow(df2)-sum(missing)),
                "not normally distributed, calculating spearman corr",
                "p = ", sw$p.value))
    method='spearman'
  } else{
    method='pearson'
    print(paste(vars_[i], sprintf("(n = %i)", nrow(df2)-sum(missing)),
                "normally distributed, calculating pearson corr",
                "p = ", sw$p.value))
  }
  
  # calculate partial correlation
  colors <- c("midnightblue", "darkred")
  for (j in range(1,length(bag))){
    current_bag <- bag[j]
    print(current_bag)
    pc <- pcor.test(x=as.numeric(df2[, current_bag]),
                    y=as.numeric(df2[, vars_[i]]),
                    z=df2[, covars],
                    method=method)
    
    # show results
    print(paste(vars_[i], 'coeff =', round(pc$estimate, 3),
                ', p =', round(pc$p.value, 4)))
    
    if (amy == ""){

    # get residuals for partial correlation plot
    df2$Y_resid<-resid(lm(df2[,vars_[i]] ~ df2[,"PTEDUCAT"]+df2[,"meanage"]+
                        df2[,"APOE4"]+df2[,"PTGENDER"]))

    df2$X_resid<-resid(lm(df2[,current_bag] ~ df2[,"PTEDUCAT"]+df2[,"meanage"] +
                         df2[,"APOE4"]+df2[,"PTGENDER"]))
    # plot if significant
    if (pc$p.value < (0.1)){
      g <- ggplot(df2) +
        geom_point(aes(x=X_resid, y=Y_resid),
                   color=colors[j], fill=colors[j],
                   alpha=0.2, size=3) + 
        stat_smooth(method=lm, fullrange=FALSE, aes(x=X_resid,
                                                    y=Y_resid),
                    linetype="dashed", color=colors[j], fill=colors[j],
                    alpha=0.2, lwd=1.2)+
        #scale_x_continuous(breaks=seq(-12,10,4)) +
        ylab(vars_[i]) +
        xlab("\nBAG [years]") +
        theme_classic() +
        scale_color_discrete() +
        theme(text = element_text(size=20))
      g
      ggsave(filename = paste(group, '_', vars_[i], '_', current_bag,
                              '_', atlas, amy, ".png", sep=""),
           path = sprintf("2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/%s",
                            database, group, long_psychpath),
             width = 10, height = 10, device='tiff', dpi=300)
    }
    
    df3 <- df2[!is.na(df2$AT),]
    # PET amyloid moderation effect?
    moderation <- lm(df3[,vars_[i]] ~ df3[,current_bag]*
                     df3$AT+
                     df3$meanage+df3$PTGENDER+df3$PTEDUCAT+df3$APOE4) # TODO: use bootstrapping?
    print(
      sprintf("Moderation of amyloid on interaction %s - %s: %s",
              current_bag, vars_[i], summary(moderation)$coefficients[8,4]))
    
    alpha <- c(
      "0"=0.2,
      "1"=0.4)
    linetype <- c(
      "0"="dashed",
      "1"="solid"
    )
    df3$Y_resid<-resid(lm(df3[,vars_[i]] ~ df3[,"PTEDUCAT"]+df3[,"meanage"]+
                           df3[,"APOE4"]+df3[,"PTGENDER"]))
    
    df3$X_resid<-resid(lm(df3[,current_bag] ~ df3[,"PTEDUCAT"]+df3[,"meanage"] +
                           df3[,"APOE4"]+df3[,"PTGENDER"]))
    if (summary(moderation)$coefficients[8,4]<.1){
      h <- ggplot(df3) +
        stat_smooth(method=lm, fullrange=FALSE, 
                    aes(x=X_resid,
                        y=Y_resid,
                        alpha=df3[, "AT"],
                        linetype=df3[, "AT"]),
                    color=colors[j], fill=colors[j],
                    alpha=0.2, lwd=1.2) +
        geom_point(aes(x=X_resid,
                       y=Y_resid,
                       alpha=df3[, "AT"],
                       shape=df3[, "AT"]),
                   color=colors[j], fill=colors[j],
                   size=3) + 
        #scale_x_continuous(breaks=seq(-12,10,4)) +
        ylab(vars_[i]) +
        xlab("\nBAG [years]") +
        theme_classic() +
        scale_color_discrete() +
        scale_alpha_manual(guide = "none",
                           values = alpha) +
        scale_linetype_manual(values = linetype) +
        theme(text = element_text(size=20))
      h
      ggsave(filename = paste(group, '_', vars_[i], '_', current_bag,
                              '_', atlas, amy, "_moderation.png", sep=""),
             path = sprintf("2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/%s",
                            database, group, long_psychpath),
             width = 12, height = 10, device='tiff', dpi=300)
}}}}}
bag <- c("MRI.BAG", "PET.BAG")
get_cors(pred_table, vars_, covars, bag)

# repeat with amyloid positives
pos_table = pred_table[pred_table$PTAU.ABETA42_0.023_cutoff==1,]
get_cors(pos_table, vars_, covars, bag, '_amypos')
neg_table = pred_table[pred_table$PTAU.ABETA42_0.023_cutoff==0,]
get_cors(neg_table, vars_, covars, bag, '_amyneg')

# DELCODE PATHOLOGY
group <- "mci"
database <- "DELCODE"
if (group == "mci"){
  pred_table <- read.csv(paste("2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/",
                               "merged_for_dx_prediction_AAL1_cropped_MCI.csv",
                               sep=""))
  pred_table$PET.BAG <- NA
} else if (group == "scd"){
  pred_table <- read.csv(paste("2_BrainAge/Brain_Age_PET_MRI/results/DELCODE/",
                               "merged_for_dx_prediction_AAL1_cropped_SMC.csv",
                               sep=""))
  pred_table$MRI.BAG <- NA
}

pred_table$sex <- ifelse(pred_table$sex=="m", 0, 1)


bag <- c("MRI.BAG")
atlas <- "AAL1_cropped"
psychpath <- "PATH"
long_psychpath <- "1_pathology"
vars_ <- c("Abeta42",
           'totaltau', 'phosphotau181', 'PTAU.ABETA42')
covars <- c('edyears', 'APOE4', 'sex')
get_cors(pred_table, vars_, covars, bag, amy="_none_")
