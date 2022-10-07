#######################################
###      CORRELATION WITH CI/NP     ###
#######################################
library(ggplot2)
library(ppcor)

rm(list=ls())
group <- 'SCD'
modality <- 'PET'
database <- 'DELCODE'
color <- ifelse(modality == 'MRI', 'midnightblue', 'darkred')
psychpath <- 'PATH'
long_psychpath <- ifelse(
  psychpath == 'PATH', '1_pathology', '1_cognitive performance')

if (database == 'ADNI'){
  pred_table <- read.csv(sprintf(
    '2_BrainAge/PET_MRI_age_final/results/ADNI/%s/pred_merged_%s_%s_%s.csv',
    group, group, modality, psychpath))
}else if (database == 'DELCODE'){
  pred_table <- read.csv(sprintf(
    '2_BrainAge/PET_MRI_age_final/results/DELCODE/%s/%s-predicted_age_%s_BAGGED.csv',
    group, modality, group))
  dem <- read.csv2(sprintf(
    '2_BrainAge/PET_MRI_age_final/data/DELCODE/%s/%s.csv', group, group),
    na.strings = "", dec = ".")
  dem <- subset(dem, select = c(PTID, ABETA, TAU, PTAU, PTEDUCAT, ApoE,
                                sex))
  dem$APOE4 <- ifelse(dem$ApoE=="02. Mrz" | dem$ApoE=="03. Mrz", 0,
                      ifelse(dem$ApoE=="03. Apr" | dem$ApoE=="02. Apr", 1, 2))
  dem$PTGENDER <- ifelse(dem$sex == "m", 0, 1)
  pred_table <- merge(pred_table, dem, by = 'PTID', all.x = TRUE, all.y = FALSE)
  pred_table$BPAD <- as.numeric(pred_table$Prediction - pred_table$Age)
}

if (psychpath == 'PATH'){
  if (database == 'ADNI'){
    vars_ <- c('AV45', 'ABETA', 'TAU', 'PTAU')
  }else  if (database == 'DELCODE'){
    vars_ <- c('ABETA', 'TAU', 'PTAU')
  }
} else{
  vars_ <- c('ADNI_MEM', 'ADNI_EF')
}

# FIND PARTIAL CORRELATIONS
for (i in 1:length(vars_)){
  # exclude missing values in var of interest
  missing <- is.na(pred_table[, vars_[i]]) | is.na(pred_table$APOE4) # pcor can't handle missing values
  
  # assess normality
  sw <- shapiro.test(pred_table[, vars_[i]][!missing])
  if (sw$p.value < 0.05){
    print(paste(vars_[i], "not normally distributed, calculating spearman corr",
                "p = ", sw$p.value))
    method='spearman'
  } else{
    method='pearson'
    print(paste(vars_[i], "normally distributed, calculating pearson corr",
                "p = ", sw$p.value))
  }
  
  # calculate partial correlation
  pc <- pcor.test(x=pred_table$BPAD[!missing],
                  y=pred_table[, vars_[i]][!missing],
                  z=pred_table[, c(
                    'PTEDUCAT', 'Age', 'APOE4', 'PTGENDER')][!missing,],
                  method=method)
  
  # show results
  print(paste(vars_[i], 'coeff =', round(pc$estimate, 3),
              ', p =', round(pc$p.value, 4)))
  
  # plot if significant
  if (pc$p.value < (0.05/length(vars_))){
    g <- ggplot(pred_table) +
      geom_point(aes(x=BPAD, y=pred_table[, vars_[i]]), color=color, fill=color,
                 alpha=0.2, size=3) + 
      stat_smooth(method=lm, fullrange=FALSE, aes(x=BPAD, y=pred_table[, vars_[i]]),
                  linetype="dashed", color=color, fill=color, alpha=0.2, lwd=1.2)+
      #scale_x_continuous(breaks=seq(-12,10,4)) +
      ylab(vars_[i]) +
      xlab("\nBAG [years]") +
      theme_classic() +
      scale_color_discrete() +
      theme(text = element_text(size=20))
    g
    ggsave(filename = paste(group, '_', vars_[i], '_', modality, ".png", sep=""),
         path = sprintf("2_BrainAge/PET_MRI_age_final/results/%s/%s/%s",
                          database, group, long_psychpath),
           width = 10, height = 10, device='tiff', dpi=300)
  }
}
