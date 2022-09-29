#######################################
###      CORRELATION WITH CI/NP     ###
#######################################
library(ggplot2)
library(ppcor)

rm(list=ls())
group <- 'MCI'
modality <- 'PET'
database <- 'ADNI'
color <- ifelse(modality == 'MRI', 'cyan4', 'coral')
psychpath <- 'PATH'
long_psychpath <- ifelse(
  psychpath == 'PATH', '1_pathology', '1_cognitive performance')

pred_table <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age0/results/ADNI/%s/pred_merged_%s_%s_%s.csv',
  group, group, modality, psychpath))

if (psychpath == 'PATH'){
  vars_ <- c('AV45', 'ABETA', 'TAU', 'PTAU')
} else{
  vars_ <- c('ADNI_MEM', 'ADNI_EF')
}

# FIND PARTIAL CORRELATIONS
for (i in 1:length(vars_)){
  # exclude missing values in var of interest
  missing <- is.na(pred_table[, vars_[i]])  # pcor can't handle missing values
  
  # calculate partial correlation
  pc <- pcor.test(x=pred_table$BPAD[!missing],
                  y=pred_table[, vars_[i]][!missing],
                  z=pred_table[, c(
                    'PTEDUCAT', 'Age', 'APOE4', 'PTGENDER')][!missing,],
                  method='spearman')
  
  # show results
  print(paste(vars_[i], 'rho =', round(pc$estimate, 3),
              ', p =', round(pc$p.value, 3)))
  
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
           path = sprintf("2_BrainAge/PET_MRI_age0/results/%s/%s/%s",
                          database, group, long_psychpath),
           width = 10, height = 10, device='tiff', dpi=300)
  }
}

