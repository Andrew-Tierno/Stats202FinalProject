source("Scripts/DateUtil.R")
library(randomForest) #Random Forest/Bagging
library(tree) #Decision Trees
library(glmnet) #Lasso/Ridge
library(MASS) #LDA/QDA
library(class) #KNN
library(e1071) #SVM/Naive Bayes

train <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")

train$relevance <- as.factor(train$relevance)

startTime <- getCurrentTime()
bag.fit <- randomForest(relevance ~ ., data=train[1:10000, ], mtry=(ncol(train) - 1), importance=T)
printElapsedTime(startTime)
predictedVals <- predict(bag.fit, train[20001:25000, ])
pctError <- round((1 - sum(predictedVals == train[20001:25000, "relevance"]) / length(predictedVals)) * 100, 2)
print(paste0("Percent Error: ", pctError, "%"), quote=F)
