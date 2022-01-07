mri <- read.csv2('C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS/OASIS_parcels_MRI.csv', stringsAsFactors = FALSE)
pet <- read.csv2('C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS/OASIS_parcels_FDG.csv', stringsAsFactors = FALSE)

pet <- subset(pet, pet$name %in% mri$name)
mri <- subset(mri, mri$name %in% pet$name)

write.csv(mri, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS/OASIS_parcels_MRI_final.csv", row.names = FALSE)
write.csv(pet, "C:/Users/doeringe/Documents/BrainAge/PET_MRI_age/data/OASIS/OASIS_parcels_PET_final.csv", row.names = FALSE)
