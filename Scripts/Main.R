train <- read.csv("Data/training.csv")
test <- read.csv("Data/test.csv")
#query_id, url_id, query_length, is_homepage, sig1, sig2,..., sig8, relevance
colnames(train)
head(train)
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
