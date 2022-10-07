########################################
# Demographic differences in brain age #
########################################

rm(list=ls())
# DEFINE
group <- 'MCI'
data <- 'ADNI'
model.add <- ifelse(group == 'CN' & data == 'ADNI', '', '_BAGGED')
pred_table_PET <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/results/%s/%s/PET-predicted_age_%s%s.csv',
  data, group, group, model.add))
pred_table_MRI <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/results/%s/%s/MRI-predicted_age_%s%s.csv',
  data, group, group, model.add))

if (data == 'ADNI'){
  dem <- read.csv2("ADNImerge/ADNIMERGE_full.csv", na.strings = "", dec=".")
  dem <- dem[dem$VISCODE == "bl",]
  dem <- subset(dem, select = c(PTID, PTEDUCAT, PTGENDER, APOE4))
} else if (data == 'OASIS'){
  dem <- read.csv2("2_BrainAge/PET_MRI_age_final/data/OASIS/OASIS_demographics_PET.csv",
                   na.strings = "", dec=".")
  dem <- subset(dem, select = c(PTID, PTEDUCAT, PTGENDER))
} else{
  dem <- read.csv2(sprintf(
    "2_BrainAge/PET_MRI_age_final/data/%s/%s/%s.csv", data, group, group),
     na.strings = "", dec=".")
  dem$PTGENDER <- ifelse(dem$sex=="m", "Male", "Female")
  dem$APOE4 <- ifelse(dem$ApoE=="02. Mrz" | dem$ApoE=="03. Mrz", 0,
                      ifelse(dem$ApoE=="03. Apr" | dem$ApoE=="02. Apr", 1, 2))
  dem <- subset(dem, select = c(PTID, PTEDUCAT, PTGENDER, APOE4, ApoE))
}



pred_table_PET <- merge(pred_table_PET, dem, by="PTID", all.x = TRUE,
                        all.y = FALSE)
pred_table_MRI <- merge(pred_table_MRI, dem, by="PTID", all.x = TRUE,
                        all.y = FALSE)

pred_table_PET$BAG <- pred_table_PET$Prediction - pred_table_PET$Age
pred_table_MRI$BAG <- pred_table_MRI$Prediction - pred_table_MRI$Age

t.test(pred_table_PET$BAG[pred_table_PET$PTGENDER == "Female"],
       pred_table_PET$BAG[pred_table_PET$PTGENDER == "Male"])
t.test(pred_table_MRI$BAG[pred_table_MRI$PTGENDER == "Female"],
       pred_table_MRI$BAG[pred_table_MRI$PTGENDER == "Male"])

t.test(pred_table_PET$BAG[pred_table_PET$APOE4 == 0],
       pred_table_PET$BAG[pred_table_PET$APOE4 > 0])
t.test(pred_table_MRI$BAG[pred_table_MRI$APOE4 == 0],
       pred_table_MRI$BAG[pred_table_MRI$APOE4 > 0])

cor.test(pred_table_PET$BAG, pred_table_PET$PTEDUCAT)
cor.test(pred_table_MRI$BAG, pred_table_MRI$PTEDUCAT)
