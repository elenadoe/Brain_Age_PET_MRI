#### Calculate performance of brain age estimation models ####
# Use all data for CN and bagged data for other samples
# OASIS results different from Python demos, because SCD are excluded in this script

rm(list=ls())
# implementation of scikit learn's r-squared in R
# scikit learn's definition of r-squared is based on Wikipedia
r2_score <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - ss_res/ss_tot
  return(r2)
}

group <- 'CN'
atlas <- 'Sch_Tian_1mm'
database <- 'OASIS'
modality <- 'PET'
model.add <- ifelse(group == 'CN' & database == 'ADNI', '', '_BAGGED')

df <- read.csv(sprintf(
  '2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/%s-predicted_age_%s_%s%s.csv',
  database, group, modality, atlas, group, model.add))
df$BAG <- df$Prediction - df$Age

if (database == 'OASIS'){
  scd_eval <- read.csv2(
    '2_BrainAge/Brain_Age_PET_MRI/data/OASIS/CN/OASIS_SCDeval_USDb9.csv')
  cn_ids <- scd_eval$name[scd_eval$group == "CN"]
  df <- df[df$PTID %in% cn_ids,]
}

n <- nrow(df)
mae <- mean(abs(df$BAG))
range <- range(df$BAG)
me <- mean(df$Prediction - df$Age)
rsquared <- r2_score(df$Age, df$Prediction)

sprintf("%s-predicted brain age in %s %s (%s, n=%s):",
        modality, database, group, atlas, n)
sprintf("MAE = %s, range = %s - %s, me = %s, R2 = %s",
        round(mae, 2) , round(range[1], 1), round(range[2], 1),
        round(me, 2), round(rsquared, 2))

df_mri <- read.csv(sprintf(
  '2_BrainAge/Brain_Age_PET_MRI/results/%s/%s/%s-predicted_age_%s_%s%s.csv',
  database, group, "MRI", atlas, group, model.add))
df_mri$BAG <- df_mri$Prediction - df_mri$Age
if (database == "OASIS"){
  df_mri <- df_mri[df_mri$PTID %in% cn_ids,]
}

t.test(abs(df$BAG), abs(df_mri$BAG))
