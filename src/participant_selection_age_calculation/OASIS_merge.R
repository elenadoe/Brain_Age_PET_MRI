participants <- read.csv2("C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/parcels.csv")
dem <- read.csv2("C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS_ADRC.csv")

# add age, ensure they're still cognitively normal at that point
participants$age <- NA
participants$daysDiffNPTesting <- NA
participants$mmse <- NA
participants$cdr <- NA
for (p in 1:nrow(participants)){
  id <- participants$ID[p]
  day_dem <- dem$dayAfterBaseline[dem$ID == id]
  day_par <- participants$dayAfterBaseline[participants$ID == id]
  
  # find closest neuropsychological evaluation to scan
  diff_test <- abs(day_dem - day_par)
  diff <- min(diff_test)
  which_diff <- which(diff_test == diff)
  
  # save info about distance to neuropsychological evaluation
  # (dropped before saving to csv)
  participants$daysDiffNPTesting[participants$ID == id] <- diff
  participants$age[participants$ID == id] <- as.numeric(dem$ageAtEntry[dem$ID == id][which_diff]) + 
                      day_par/365
  # save mmse and cdr scores
  participants$mmse[participants$ID == id] <- as.numeric(dem$mmse[dem$ID == id][which_diff])
  participants$cdr[participants$ID == id] <- as.numeric(dem$cdr[dem$ID == id][which_diff])
}

# only keep participants who are cognitively normal according to ADNI criteria
# MMSE > 24, CDR = 0
participants_mri <- participants[(participants$mmse>24) & (participants$cdr==0),]

# remove IDs from PET who are not cognitively normal at mri scan date and vice versa
participants_pet <- read.csv2("C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/parcels_FDG_fdg.csv")
participants_pet <- participants_pet[participants_pet$ID %in% participants_mri$ID,]
participants_mri <- participants_mri[participants_mri$ID %in% participants_pet$ID,]

# remove unnecessary variables 2.0
participants_mri <- subset(participants_mri, select=-c(mmse, cdr, 
                                                      daysDiffNPTesting, ID, 
                                                      dayAfterBaseline))
participants_pet <- subset(participants_pet, select=ID)

write.csv(participants_mri, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS_CN_clean_MRI.csv", row.names = FALSE)
write.csv(participants_pet, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS_CN_clean_PET.csv", row.names = FALSE)
