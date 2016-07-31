source("Scripts/Main.r")

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

attach(train)

out <- plotRFCost()
write.csv(out, "Data/RFOut.csv")
plot(out[2:nrow(out), ])