source("Scripts/Main.r")

trainMaster <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")
trainMaster$relevance <- as.factor(trainMaster$relevance)
trainMaster$is_homepage <- as.factor(trainMaster$is_homepage)
trainMaster <- trainMaster[,c("query_length", "is_homepage", "sig1", "sig2", "sig3", "sig4", "sig5", "sig6", "sig7", "sig8", "url_id", "relevance")]
trainPCA <- trainMaster[c("url_id", "query_length", "sig1", "sig2", "sig3", "sig4", "sig5", "sig6", "sig7", "sig8")]
trainPCA <- scaleDF(trainPCA)
trainPCAMod <- trainPCA
for(i in 1:ncol(trainPCA)) {
  for(j in i:ncol(trainPCA)) {
    trainPCAMod[,paste0("sig", i, j)] <- trainPCA[,i] * trainPCA[,j]
    trainPCAMod[,paste0("sig", i, j)] <- (trainPCAMod[,paste0("sig", i, j)] - mean(trainPCAMod[,paste0("sig", i, j)])) / sd(trainPCAMod[,paste0("sig", i, j)])
  }
}

pca.fit <- prcomp(trainPCAMod)
trainMasterPCA <- as.data.frame(as.matrix(trainPCAMod) %*% loadings(pca.fit)[,1:6])
trainMaster <- cbind(trainMasterPCA, trainMaster[,c("is_homepage", "relevance")])
colVec <- ifelse(trainMaster$relevance == 1, "green", "red")
plot(trainMaster$PC1, trainMaster$PC2, col=colVec)
#Random seed, value is arbitrary
SEED <- 42
#Number of entries to use from trainMaster, since trainMaster is
#too large for experimenting.
NUM_ENTRIES <- nrow(trainMaster)
#Number of folds for k-folds CV. Each fold will contain
#NUM_ENTRIES / NUM_FOLDS entries.
NUM_FOLDS <- 10


set.seed(SEED)
train <- trainMaster[sample(1:nrow(trainMaster), NUM_ENTRIES), ]
#Sample uses uniform distributions so it will produce NUM_FOLDS
#folds that are rougly identical in size
folds <- sample(1:NUM_FOLDS, nrow(train), replace = T)

attach(train)
kError <- c()
for(i in 26:50) {
  print(paste0("Testing k=", i), quote=F)
  knn.fit <- knn(subset(train[folds!=1, ], select=-c(relevance)), subset(train[folds==1, ], select=-c(relevance)), train[folds!=1, "relevance"], k=i)
  kError[length(kError) + 1] <- 1 - sum(knn.fit == train[folds==1, "relevance"]) / length(knn.fit)
}

#out <- plotRFCost()
#write.csv(out, file="Data/RFPlot.csv", row.names = F)