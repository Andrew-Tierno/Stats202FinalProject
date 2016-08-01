source("Scripts/DateUtil.R")
library(randomForest) #Random Forest/Bagging
library(tree) #Decision Trees
library(glmnet) #Lasso/Ridge
library(MASS) #LDA/QDA
library(class) #KNN
library(e1071) #SVM/Naive Bayes
library(pls) #PCR/PLSR
library(beepr) #Announces when processes finish

#N=20000, k=10, % Error=34.72%, total time=9m31s, avg time=57s
#N=40000, k=10, % Error=34.65%, total time=33m02s, avg time=3m18s
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
testSVM <- function(costVal, kern = "radial") {
  print("========Testing SVM========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    svm.fit <- svm(relevance ~ ., data=train[folds != i, ], cost = costVal, kernel = kern)
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

#Minimum for cost=100 with % error of 32.44% on N=20000, k=10
plotSVMCost <- function() {
  print("========Sweep SVM========", quote=F)
  costs <- c(.1, .25, .5, .75, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
  out <- data.frame(cost=0, error=0)
  for(i in 1:length(costs)) {
    print(paste0("Testing cost ", i, " of ", length(costs), "."), quote=F)
    totalError <- 0
    totalStartTime <- getCurrentTime()
    for(j in 1:NUM_FOLDS) {
      startTime <- getCurrentTime()
      print(paste0("Testing fold ", j, " of ", NUM_FOLDS, "."), quote=F)
      svm.fit <- svm(relevance ~ ., data=train[folds != i, ], cost = costs[i])
      predictedVals <- predict(svm.fit, train[folds == i, ])
      pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
      totalError <- pctError + totalError
      printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
      print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
    }
    out <- rbind(out, data.frame(cost=costs[i], error=totalError/NUM_FOLDS))
    printElapsedTime(totalStartTime, "Total Elapsed Time: ")
    print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
    print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  }
  beep("complete")
  return(out)
}

plotRFCost <- function() {
  print("========Sweep RF========", quote=F)
  mList <- 1:(ncol(train) - 1)
  out <- data.frame(m=0, error=0)
  for(i in 1:length(mList)) {
    print(paste0("Testing m ", i, " of ", length(mList), "."), quote=F)
    totalError <- 0
    totalStartTime <- getCurrentTime()
    for(j in 1:NUM_FOLDS) {
      startTime <- getCurrentTime()
      print(paste0("Testing fold ", j, " of ", NUM_FOLDS, "."), quote=F)
      rf.fit <- randomForest(relevance ~ ., data=train[folds != i, ], mtry = mList[i])
      predictedVals <- predict(rf.fit, train[folds == i, ])
      pctError <- round((1 - sum(predictedVals == train[folds == i, "relevance"]) / length(predictedVals)) * 100, 2)
      totalError <- pctError + totalError
      printElapsedTime(startTime, paste0("Elapsed Time for Fold ", i, ": "))
      print(paste0("Percent Error for Fold ", i, ": ", pctError, "%"), quote=F)
    }
    out <- rbind(out, data.frame(m=mList[i], error=totalError/NUM_FOLDS))
    printElapsedTime(totalStartTime, "Total Elapsed Time: ")
    print(paste0("Average Elapsed Time: ", getElapsedTime(totalStartTime) / NUM_FOLDS, "s"), quote=F)
    print(paste0("Average Percent Error: ", round(totalError/NUM_FOLDS, 2), "%"), quote=F)
  }
  beep("complete")
  return(out)
}

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

testLR <- function() {
  print("========Testing LR========", quote=F)
  totalError <- 0
  totalStartTime <- getCurrentTime()
  for (i in 1:NUM_FOLDS) {
    startTime <- getCurrentTime()
    glm.fit <- glm(relevance ~ ., data=train[folds != i, ], family = binomial)
    predictedProbs <- predict(glm.fit, train[folds == i, ])
    predictedVals <- rep(factor(1, levels=c(0, 1)), length(predictedProbs))
    predictedVals[predictedProbs < 0] <- factor(0)
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

scaleDF <- function(df) {
  for(i in 1:ncol(df)) {
    df[,i] <- (df[,i] - mean(df[,i])) / sd(df[,i])
  }
  return(df)
}