data <- read.csv2("Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/parcels_all.csv", sep = ";")
idsused <- read.csv2("Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/ADNI_merge_Amyloid_3.csv")

df <- data.frame(matrix(NA,nrow = length(idsused$ID), 
                        ncol = length(unique(data$ROI.name))+1))
colnames(df) <- c("ID", unique(data$ROI.name))
df$ID <- idsused$PTID

for (i in 1:nrow(data)){
  roi <- data$ROI.name[i]
  value <- data$Average[i]
  id <- data$ID[i]
  df[roi][df$ID == id,] <- value
}

dev.new(width=20,height=10, unit = "in", noRStudioGD = TRUE)

par(mfrow=c(2,5))

for (i in 2:11){
  scatter.smooth(df[i], ylab = c(colnames(df[i]), "SUVR"), ylim = c(5000, 30000))
}
dev.new(4,5)
par(mar=c(12,4,2,2)+.1)
boxplot(subset(data,select=-ID), las = 3, par(cex = 0.6), ylab = "non-standardized voxel intensity")
all <- subset(df,select='Cerebellum Anterior Lobe ')
which(list(all) %in% list(boxplot(all)$out))
boxplot(subset(df,select='Cerebellum Anterior Lobe '))$out

boxplot(subset(df,select='Cerebellum Anterior Lobe '))

boxplot(subset(all))$out
which(all == 1.235200)
