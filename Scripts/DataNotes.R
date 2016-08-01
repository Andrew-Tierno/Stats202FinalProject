trainMaster <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")

trainMaster$relevance <- as.factor(trainMaster$relevance)
trainMaster$is_homepage <- as.factor(trainMaster$is_homepage)

#query_id, url_id, query_length, is_homepage, sig1, sig2,..., sig8, relevance
colnames(train)
head(train)

train$is_homepage <- as.logical(train$is_homepage)
train$relevance <- as.logical(train$relevance)

library(plyr)
attach(train)
length(url_id) #80046
length(unique(url_id)) #75231
range(sig1) #[0, 1]
range(sig2) #[0, 0.86]
range(sig3) #[0, 673637]
range(sig4) #[0, 660939]
range(sig5) #[0, 46994]
range(sig6) #[0, 3645]
range(sig7) #[0, 0.88]
range(sig8) #[0, 0.94]

#Appears left skewed
hist(sig1)
#Appears normally distributed
hist(sig2)
#Appears normally distributed
hist(sig7)
#Appears skewed right
hist(sig8)

hist(sig6)
#50% of values are <417, 90% are <8489, and 99% are <86749 (max 673637).
#1130 observations (1.66%) have a sig3 that is zero
quantile(sig3, probs=c(.50, .90, .99)) 
#50% of values are <220, 90% are <1437, and 99% are <8186 (max 660939).
#5794 observations (7.24%) have a sig4 that is zero
quantile(sig4, probs=c(.50, .90, .99)) 
#50% of values are <64, 90% are <1238, and 99% are <8129 (max 46994).
#9046 observations (11.3%) have a sig5 that is zero
quantile(sig5, probs=c(.50, .90, .99)) 
#50% of values are <0, 90% are <18, and 99% are <293 (max 3645).
#52165 observations (65.2%) have a sig6 that is zero.
quantile(sig6, probs=c(.50, .90, .99)) 

sum(is_homepage & relevance) / length(relevance) #13.08%
sum(is_homepage & !relevance) / length(relevance) #13.81%
sum(!is_homepage & relevance) / length(relevance) #30.63%
sum(!is_homepage & !relevance) / length(relevance) #42.48%

pca <- princomp(train)
weightMat <- loadings(pca)

trainMaster <- read.csv("Data/training.csv")
attach(trainMaster)
trainCor <- trainMaster[ ,c("query_length", "sig1", "sig2", "sig4L", "sig6L", "sig7", "sig8")]
minCorAbs <- Inf
minA <- 0
dataPoints <- data.frame(aVal=0, absVal=0)
for (a in seq(0, pi, by=0.001)) {
  currCorAbs <- 0
  sig35L <- sin(a) * sig3L + cos(a) * sig5L
  for(i in 1:ncol(trainCor)) {
    currCorAbs <- currCorAbs + cor(sig35L, trainCor[,i])^2
  }
  dataPoints <- rbind(dataPoints, data.frame(aVal=a, absVal=currCorAbs))
  if(currCorAbs < minCorAbs) { 
    minCorAbs <- currCorAbs
    minA <- a
  }
}
dataPoints <- dataPoints[2:nrow(dataPoints), ]
plot(dataPoints, xlab="a", ylab="Total Squared Correlation", main="Total Sqd. Correlation for sig3L sin(a) + sig5L cos(a)", type="l")

attach(trainMaster)
propMat <- as.matrix(data.frame(homepage=c(sum(is_homepage == 1 & relevance == 1), sum(is_homepage == 1 & relevance != 1)), nothomepage=c(sum(is_homepage == 0 & relevance == 1), sum(is_homepage == 0 & relevance != 1))))
prop.test(propMat)
propMat <- as.matrix(data.frame(homepage=c(sum(sig6 == 0 & relevance == 1), sum(sig6 == 0 & relevance != 1)), nothomepage=c(sum(sig6 != 0 & relevance == 1), sum(sig6 != 0 & relevance != 1))))
prop.test(propMat)

trainPCA <- trainMaster[c("url_id", "query_length", "sig1", "sig2", "sig3", "sig4", "sig5", "sig6", "sig7", "sig8")]
trainPCA <- scaleDF(trainPCA)
trainPCAMod <- trainPCA
for(i in 1:ncol(trainPCA)) {
  for(j in i:ncol(trainPCA)) {
    trainPCAMod[,paste0("sig", i, j)] <- trainPCA[,i] * trainPCA[,j]
  }
}
trainPCAMod$sig0 <- rep(1 / sqrt(nrow(trainPCAMod)), nrow(trainPCAMod))
pca.fit <- prcomp(trainPCAMod)

scaleDF <- function(df) {
  for(i in 1:ncol(df)) {
    df[,i] <- (df[,i] - mean(df[,i])) / sd(df[,i])
  }
  return(df)
}
