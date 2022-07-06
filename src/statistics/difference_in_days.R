rm(list=ls())
mri <- read.csv("2_BrainAge/PET_MRI_age/results/MCI/MRI-predicted_age_MCI_0.csv")
pet <- read.csv("2_BrainAge/PET_MRI_age/results/MCI/PET-predicted_age_MCI_0.csv")
mri_sess <- read.csv2("2_BrainAge/PET_MRI_age/data/MCI/MCI_MRI_parcels.csv")
pet_sess <- read.csv2("2_BrainAge/PET_MRI_age/data/MCI/MCI_PET_parcels.csv")
mri$mri_sess_date <- NA
pet$pet_sess_date <- NA
for (i in 1:nrow(mri_sess)){
  mri_sess$mri_sess_date[i] <- paste(substr(mri_sess$sess[i],11,12),
                                ".", substr(mri_sess$sess[i],9,10),
                                ".", substr(mri_sess$sess[i],5,8),
                                sep = "")
  mri_sess$mri_sess_date[i] <- as.Date(mri_sess$mri_sess_date[i],
                                       format = "%d.%m.%Y")
}

for (i in 1:nrow(pet_sess)){
  pet_sess$pet_sess_date[i] <- paste(substr(pet_sess$sess[i],5,6),
                                ".", substr(pet_sess$sess[i],7,8),
                                ".", substr(pet_sess$sess[i],1,4),
                                sep = "")
  pet_sess$pet_sess_date[i] <- as.Date(pet_sess$pet_sess_date[i],
                                       format = "%d.%m.%Y")
}

pet_sess <- subset(pet_sess, select = c(name, sess, pet_sess_date))
mri_sess <- subset(mri_sess, select = c(name, sess, mri_sess_date))
sess <- merge(pet_sess, mri_sess, by = "name")
sess$diffdays <-  as.numeric(sess$pet_sess_date) -  as.numeric(sess$mri_sess_date)
sess$absdiff <- abs(sess$diffdays)
mci <- subset(mri, select = (PTID))
mci$name <- mci$PTID
mci <- merge(mci, sess, all.x = T, all.y = F, by = "name")
# systematic difference of PET and MRI days
table(mci$diffdays>0)

# mean and std of difference in days
mean(mci$absdiff)
sd(mci$absdiff)
