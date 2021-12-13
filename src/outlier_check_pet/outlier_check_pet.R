rm(list=ls())
data.pet <- read.csv2("BrainAge/PET_MRI_age/data/ADNI/PET_parcels_withNP.csv",
                      dec = ".")
data.mri <- read.csv2("BrainAge/PET_MRI_age/data/ADNI/MRI_parcels_withNP.csv",
                      dec = ".")

col <- colnames(data.pet)[4:219]
ids <- data.pet$name
iqr_val <- 3

outliers.pet <- list()
outlier_hubs.pet <- data.frame(matrix(NA,nrow = length(col)))
outlier_hubs.pet$ROI <- col
outliers.mri <- list()
outlier_hubs.mri <- data.frame(matrix(NA,nrow = length(col)))
outlier_hubs.mri$ROI <- col
for (i in 1:length(col)){
  d.pet <- subset(data.pet, select=col[i])
  b.pet <- boxplot(d.pet, range = iqr_val)
  roi_outliers <- length(which(t(d.pet) %in% b.pet$out))
  outlier_hubs.pet$n[outlier_hubs.pet$ROI == col[i]] <- roi_outliers
  outliers.pet <- c(outliers.pet, ids[which(t(d.pet) %in% b.pet$out)])
  
  d.mri <- subset(data.mri, select=col[i])
  b.mri <- boxplot(d.mri, range = iqr_val)
  roi_outliers <- length(which(t(d.mri) %in% b.mri$out))
  outlier_hubs.mri$n[outlier_hubs.mri$ROI == col[i]] <- roi_outliers
  outliers.mri <- c(outliers.mri, ids[which(t(d.mri) %in% b.mri$out)])
}

print(c("Outliers in PET: ", length(unique(outliers.pet)), 
        "Outliers in MRI: ", length(unique(outliers.mri))))

dev.new(width=20,height=10, unit = 'in', noRStudioGD = TRUE)
par(mar=c(9,4,2,10)+.1)
imp_ROI.pet <- list()
imp_ind.pet <- which(outlier_hubs.pet$n>0)
imp_ROI.mri <- list()
imp_ind.mri <- which(outlier_hubs.mri$n>0)
for (i in 1:length(col)){
  if (i %in% imp_ind.pet){
    imp_ROI.pet <- c(imp_ROI.pet, outlier_hubs.pet$ROI[i])
  } else {imp_ROI.pet <- c(imp_ROI.pet,"")}
  if (i %in% imp_ind.mri){
    imp_ROI.mri <- c(imp_ROI.mri, outlier_hubs.mri$ROI[i])
  } else {imp_ROI.mri <- c(imp_ROI.mri,"")}
}
par(family="serif")
barplot(outlier_hubs.pet$n, ylim = c(0,15),
        names.arg = imp_ROI.pet, las = 2, cex.names = 0.51,
        cex.lab = 0.8,
        xlab = "", ylab = "Number of Participants")
barplot(outlier_hubs.mri$n, ylim = c(0,15),
        names.arg = imp_ROI.mri, las = 2, cex.names = 0.51,
        cex.lab = 0.8,
        xlab = "", ylab = "Number of Participants")

sum(unique(outliers.mri) %in% unique(outliers.pet))

pet_wo_outliers <- data.pet[!(data.pet$name %in% outliers.pet) &
                            !(data.pet$name %in% outliers.mri),]
# these should be FALSE
any(pet_wo_outliers$name %in% outliers.pet)
any(pet_wo_outliers$name %in% outliers.mri)

mri_wo_outliers <- data.mri[!(data.mri$name %in% outliers.pet) &
                            !(data.mri$name %in% outliers.mri),]
# these should be FALSE
any(mri_wo_outliers$name %in% outliers.pet)
any(mri_wo_outliers$name %in% outliers.mri)
# these should be TRUE
nrow(mri_wo_outliers) == nrow(pet_wo_outliers)
all(mri_wo_outliers$name == pet_wo_outliers$name)

write.csv(pet_wo_outliers,"BrainAge/PET_MRI_age/data/ADNI/PET_parcels_withNP_nooutliers.csv",
          row.names = FALSE)
write.csv(mri_wo_outliers,"BrainAge/PET_MRI_age/data/ADNI/MRI_parcels_withNP_nooutliers.csv",
          row.names = FALSE)
