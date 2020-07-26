library(dplyr)
library(data.table)
library(astsa)
library(xgboost)
library(caret)
library(tidyverse)
library(gbm)
library(iml)
library(ggplot2)
library(party)
library(kernlab)
library(bartMachine)
options(scipen=999)

#Parallel Processing
library(doParallel)
cl <- makeCluster(2)
registerDoParallel(cl)
options(java.parameters = "-Xmx2g")


#load city
munici <- read.csv("Cartagena.csv", sep = ",")

#remove NAs
munici <- munici[-(1:4), ]
str(munici)

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici$Attentions, p=0.7, list = F)  # 70% training data
trainData <- munici[trainDataIndex, ]
testData <- munici[-trainDataIndex, ]

# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

#  xgboost model with caret
set.seed(99)  # for reproducibility
ml_model = caret::train(Attentions ~ ., data = trainData, 
                             method = "xgbTree",
                             metric = "Rsquared",
                             prob.model = TRUE, na.action = na.omit, 
                             trControl = xgb_trcontrol,
                             tuneGrid = xgbGrid)

gbm.pred <- predict(ml_model, testData, na.action = na.pass)
summary(lm(testData$Attentions ~ gbm.pred))$r.squared
plot(testData$Attentions, gbm.pred)


# Random forres model
mtry <- ncol(munici)
tunegrid <- expand.grid(.mtry=mtry)

set.seed(99)  # for reproducibility
ml_model = caret::train(Attentions  ~ ., data = trainData, 
                        method = "rf",
                        metric = "Rsquared",
                        na.action = na.omit, 
                        trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE),
                        tuneGrid = tunegrid)

gbm.pred <- predict(ml_model, testData, na.action = na.pass)
summary(lm(testData$Attentions ~ gbm.pred))$r.squared
plot(testData$Attentions, gbm.pred)


# SVM
set.seed(99)  # for reproducibility
ml_model = caret::train(Attentions  ~ ., data = trainData, 
                        method = "svmRadialCost",
                        metric = "Rsquared",
                        na.action = na.omit, 
                        trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE),
                        tuneGrid = data.frame(C = c(0.25,0.5,1,2,4,8,16,32,64,128)))

gbm.pred <- predict(ml_model, testData, na.action = na.pass)
summary(lm(testData$Attentions ~ gbm.pred))$r.squared
plot(testData$Attentions, gbm.pred)


# Bayesian Additive Regression Trees
bartGrid <- expand.grid(num_trees = c(10, 15, 20, 100), k = 2, alpha = 0.95, beta = 2, nu = 3)
set.seed(99)  # for reproducibility
ml_model = caret::train(Attentions  ~ . , data = trainData, 
                        method = "bartMachine",
                        metric = "Rsquared",
                        tuneGrid = bartGrid, 
                        na.action = na.omit, serialize = TRUE,
                        trControl = trainControl(method = "cv", number = 10))

gbm.pred <- predict(ml_model, testData, na.action = na.pass)
summary(lm(testData$Attentions ~ gbm.pred))$r.squared
plot(testData$Attentions, gbm.pred)



# performance metrics
X <- munici%>%
  dplyr::select(- Attentions) %>%
  as.data.frame()

library("future")
library("future.callr")
# Creates a PSOCK cluster with 2 cores
plan("callr", workers = 2)

# Permutation Feature Importance
predictor_imp <- Predictor$new(ml_model, data = X, y = munici$Attentions)
importace_rf = FeatureImp$new(predictor_imp, loss = "mse", n.repetitions = 500)
plot(importace_rf) + theme_bw() + theme(plot.title = element_text(size=22)) + labs(title="i")

# Feature interactions
predictor_inter <- Predictor$new(ml_model, data = X)
interactions <- Interaction$new(predictor_inter)
plot(interactions) + theme_bw() + theme(plot.title = element_text(size=22)) + labs(title="j")









