rm(list=ls())

library(dplyr)

data.pet <- read.csv("BrainAge/PET_MRI_age/data/merged/test_train_PET_42.csv",
                      dec = ".")
data.mri <- read.csv("BrainAge/PET_MRI_age/data/merged/test_train_MRI_42.csv",
                      dec = ".")

data.pet.train <- data.pet %>% filter(train == "True")
data.mri.train <- data.mri %>% filter(train == "True")

col <- colnames(data.pet.train)[5:220]
ids <- data.pet.train$name
iqr_val <- 3

# manually re-create 3*IQR range from train set
# for application to test set

dn.pet <- data.frame(lapply(subset(data.pet.train, select=col),quantile,probs=c(0.25,0.75)))
dn.pet["IQR",] <- lapply(subset(data.pet.train, select=col),IQR)
dn.pet["upperbound",] <- dn.pet["75%",]+(iqr_val*dn.pet["IQR",])
dn.pet["lowerbound",] <- dn.pet["25%",]-(iqr_val*dn.pet["IQR",])

dn.mri <- data.frame(lapply(subset(data.mri.train, select=col),quantile,probs=c(0.25,0.75)))
dn.mri["IQR",] <- lapply(subset(data.mri.train, select=col),IQR)
dn.mri["upperbound",] <- dn.mri["75%",]+(iqr_val*dn.mri["IQR",])
dn.mri["lowerbound",] <- dn.mri["25%",]-(iqr_val*dn.mri["IQR",])

outliers.pet <- list()
regions.pet <- list()
outliers.mri <- list()
regions.mri <- list()

# OUTLIER DETECTION\nIDs that appear are outside of 3*IQR
for (i in 1:length(col)){
  d.pet <- subset(data.pet, select=col[i])
  test.pet <- which((d.pet < dn.pet["lowerbound", col[i]]) | d.pet > dn.pet["upperbound", col[i]])
  d.mri <- subset(data.mri, select=col[i])
  test.mri <- which((d.mri < dn.mri["lowerbound", col[i]]) | d.mri > dn.mri["upperbound", col[i]])
  if (length(test.pet)>0){
    outliers.pet <- c(outliers.pet, data.pet$name[which((d.pet < dn.pet["lowerbound", col[i]]) | d.pet > dn.pet["upperbound", col[i]])])
    regions.pet <- c(regions.pet, col[i])
  }
  if (length(test.mri)>0){
    outliers.mri <- c(outliers.mri, data.mri$name[which((d.mri < dn.mri["lowerbound", col[i]]) | d.mri > dn.mri["upperbound", col[i]])])
    regions.mri <- c(regions.mri, col[i])
  }
}

print(c("Outliers in PET: ", length(unique(outliers.pet)), 
        "Outliers in MRI: ", length(unique(outliers.mri))))

sum(unique(outliers.mri) %in% unique(outliers.pet))

# apply ranges to test data
data.pet$IQR <- NA
data.mri$IQR <- NA

data.pet$IQR <- !(data.pet$name %in% outliers.mri) & !(data.pet$name %in% outliers.pet)
data.mri$IQR <- !(data.mri$name %in% outliers.mri) & !(data.mri$name %in% outliers.pet)


write.csv(data.pet,"BrainAge/PET_MRI_age/data/merged/test_train_PET_final.csv",
          row.names = FALSE)
write.csv(data.mri,"BrainAge/PET_MRI_age/data/merged/test_train_MRI_final.csv",
          row.names = FALSE)
