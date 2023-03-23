##########################################
#    EVALUATE BRAIN AGE                  #
##########################################

rm(list=ls())
# DEFINE
modality <- 'PET'
pred_table <- read.csv(sprintf(
    '2_BrainAge/PET_MRI_age_final/results/ADNI/CN/%s-predicted_age_CN.csv', modality))

pred_table$error <- pred_table$Prediction - pred_table$Age
sprintf('MAE: %f, MDiff: %f', mean(abs(pred_table$error)),
        mean(pred_table$error))

# COMPARE CN AND SCI BRAIN AGE
# load data
sci_dx <- read.csv(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/CN/FDG_BASELINE_HEALTHY_4_15_2021_unique.csv')
sci_dx <- sci_dx[sci_dx$Subject %in% pred_table$PTID,]

colnames(sci_dx)[which(names(sci_dx) == 'Subject')] <- 'PTID'
table(sci_dx$Visit)

# exclude visit == 2 (ADNI1): no difference was made between CN and SCI
sci_dx <- sci_dx[sci_dx$Visit != 2,]

sci_dx <- subset(sci_dx, select = c(PTID, Group))
df <- merge(pred_table, sci_dx, all.x=TRUE, by='PTID')
df$BAG <- df$Prediction - df$Age

t.test(df$Age[df$Group=='CN'], df$Age[df$Group=='SMC'])
t.test(df$BAG[df$Group=='CN'], df$BAG[df$Group=='SMC'])
table(df$Group)
