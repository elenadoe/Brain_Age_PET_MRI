# todo compile again
rm(list=ls())
group <- 'MCI'
mri <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/%s/MRI_parcels_%s_ADNI.csv',
  group, group))
pet <- read.csv(sprintf(
  '2_BrainAge/PET_MRI_age_final/data/ADNI/%s/PET_parcels_%s_ADNI.csv',
  group, group))
mri$sess <- substr(mri$sess, 5, 12)
mri <- mri[mri$age >= 60,]
pet <- pet[pet$age >= 60,]
mri$mri_sess_date <- NA
pet$pet_sess_date <- NA
for (i in 1:nrow(mri)){
  mri$mri_sess_date[i] <- paste(substr(mri$sess[i],7,8),
                                ".", substr(mri$sess[i],5,6),
                                ".", substr(mri$sess[i],1,4),
                                sep = "")
  mri$mri_sess_date[i] <- as.Date(mri$mri_sess_date[i],
                                       format = "%d.%m.%Y")
}

for (i in 1:nrow(pet)){
  pet$pet_sess_date[i] <- paste(substr(pet$sess[i],5,6),
                                ".", substr(pet$sess[i],7,8),
                                ".", substr(pet$sess[i],1,4),
                                sep = "")
  pet$pet_sess_date[i] <- as.Date(pet$pet_sess_date[i],
                                       format = "%d.%m.%Y")
}

pet <- subset(pet, select = c(name, sess, pet_sess_date))
mri <- subset(mri, select = c(name, sess, mri_sess_date))
sess <- merge(pet, mri, by = "name")
sess$diffdays <-  as.numeric(sess$pet_sess_date) -  as.numeric(sess$mri_sess_date)
sess$absdiff <- abs(sess$diffdays)

# systematic difference of PET and MRI days
table(sess$diffdays>0)

# mean and std of difference in days
mean(sess$absdiff)
sd(sess$absdiff)
