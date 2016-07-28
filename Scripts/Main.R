source("Scripts/DateUtil.R")
library(randomForest) #Random Forest/Bagging
library(tree) #Decision Trees
library(glmnet) #Lasso/Ridge
library(MASS) #LDA/QDA
library(class) #KNN
library(e1071) #SVM/Naive Bayes

trainMaster <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")

trainMaster$relevance <- as.factor(trainMaster$relevance)
trainMaster$is_homepage <- as.factor(trainMaster$is_homepage)

#Random seed, value is arbitrary
SEED <- 42
#Number of entries to use from trainMaster, since trainMaster is
#too large for experimenting.
NUM_ENTRIES <- 20000
#Number of folds for k-folds CV. Each fold will contain
#NUM_ENTRIES / NUM_FOLDS entries.
NUM_FOLDS <- 10


set.seed(SEED)
train <- trainMaster[sample(1:nrow(trainMaster), NUM_ENTRIES), ]
#Sample uses uniform distributions so it will produce NUM_FOLDS
#folds that are rougly identical in size
folds <- sample(1:NUM_FOLDS, nrow(train), replace = T)
train <- data.frame(train, folds)

#N=20000, k=10, % Error=34.77%, total time=8m58s, avg time=54s
testBagging <- function() {
  print("========Testing Bagging========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    bag.fit <- randomForest(relevance ~ ., data=train[train$folds != i, ], mtry=(ncol(train) - 1), importance=T)
    predictedVals <- predict(bag.fit, train[train$folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[train$folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
}

#N=20000, k=10, % Error=34.15%, total time=8m31s, avg time=51s
testRandomForest <- function(m = sqrt(ncol(train) - 1)) {
  print("========Testing Random Forest========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    rf.fit <- randomForest(relevance ~ ., data=train[train$folds != i, ], mtry=m, importance=T)
    predictedVals <- predict(rf.fit, train[train$folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[train$folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
}

#N=20000, k=10, % Error=34.75%, total time=7m51s, avg time=47s 
testSVM <- function() {
  print("========Testing SVM========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    svm.fit <- svm(relevance ~ ., data=train[train$folds != i, ])
    predictedVals <- predict(svm.fit, train[train$folds == i, ])
    pctError <- round((1 - sum(predictedVals == train[train$folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
    totalError <- totalError + pctError
    printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
    print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
  }
  printElapsedTime(totalStartTime, "Total Elapsed Time: ")
  print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
  print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
}