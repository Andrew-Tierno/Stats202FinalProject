source("Scripts/DateUtil.R")
library(randomForest) #Random Forest/Bagging
library(tree) #Decision Trees
library(glmnet) #Lasso/Ridge
library(MASS) #LDA/QDA
library(class) #KNN
library(e1071) #SVM/Naive Bayes
library(pls) #PCR/PLSR
library(beepr) #Announces when processes finish

trainMaster <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")

trainMaster$relevance <- as.factor(trainMaster$relevance)
trainMaster$is_homepage <- as.factor(trainMaster$is_homepage)

#Random seed, value is arbitrary
SEED <- 42
#Number of entries to use from trainMaster, since trainMaster is
#too large for experimenting.
NUM_ENTRIES <- 15000
#Number of folds for k-folds CV. Each fold will contain
#NUM_ENTRIES / NUM_FOLDS entries.
NUM_FOLDS <- 10


set.seed(SEED)
train <- trainMaster[sample(1:nrow(trainMaster), NUM_ENTRIES), ]
#Sample uses uniform distributions so it will produce NUM_FOLDS
#folds that are rougly identical in size
folds <- sample(1:NUM_FOLDS, nrow(train), replace = T)

attach(train)

#N=20000, k=10, % Error=34.72%, total time=9m31s, avg time=57s
testBagging <- function() {
  print("========Testing Bagging========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    bag.fit <- randomForest(relevance ~ ., data=train[folds != i, ], mtry=(ncol(train) - 1), importance=T)
    predictedVals <- predict(bag.fit, train[folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  beep("complete")
}

#N=20000, k=10, % Error=34.22%, total time=7m54s, avg time=47s
testRandomForest <- function(m = sqrt(ncol(train) - 1)) {
  print("========Testing Random Forest========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    rf.fit <- randomForest(relevance ~ ., data=train[folds != i, ], mtry=m, importance=T)
    predictedVals <- predict(rf.fit, train[folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  beep("complete")
}

#N=20000, k=10, % Error=34.61%, total time=6m45s, avg time=41s 
testSVM <- function(costVal) {
  print("========Testing SVM========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    svm.fit <- svm(relevance ~ ., data=train[folds != i, ], cost = costVal)
    predictedVals <- predict(svm.fit, train[folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  beep("complete")
}

plotSVMCost <- function() {
  costs <- c(.1, .5, 1, 10, 25, 50, 75, 100, 200)
  out <- data.frame(cost=0, error=0)
  for(i in 1:length(costs)) {
    print(paste0("Testing cost ", i, " of ", length(costs), "."), quote=F)
    totalError <- 0
    for(j in 1:NUM_FOLDS) {
      print(paste0("Testing fold ", j, " of ", NUM_FOLDS, "."), quote=F)
      svm.fit <- svm(relevance ~ ., data=train[folds != i, ], cost = costs[i])
      predictedVals <- predict(svm.fit, train[folds == i, ])
      pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
      totalError <- pctError + totalError
    }
    out <- rbind(out, data.frame(cost=costs[i], error=totalError/NUM_FOLDS))
  }
  beep("complete")
  return(out)
}

costPoints <- plotSVMCost()

testPLSR <- function() {
  print("========Testing PLSR========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    plsr.fit <- plsr(relevance ~ ., data=train[folds != i, ])
    predictedVals <- predict(.fit, train[folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  beep("complete")
}