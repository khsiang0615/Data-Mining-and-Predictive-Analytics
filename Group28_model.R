#load libraries
library(tidyverse)
library(caret)
library(stringi)
library(tree)
library(class)
library(ROCR)
library(glmnet)
library(rpart)
library(ranger)
library(gbm)
library(xgboost)
library(ggplot2)

#Set random seed
set.seed(1)

#===========load data===========

#dummy data for tree, ridge, and boosting
dummy_labeled_data <- read_csv("labeled_data.csv")
dummy_unlabeled_data <- read_csv("unlabeled_data.csv")

#load clean data that didn't convert variables into dummies
tr <- read_csv("no_dummy.csv")
tr_y <- read_csv("train_y.csv")

#filter out the training set and validation set
clean_labeled <- tr %>%
  filter(is_train == 1)

clean_unlabeled <- tr %>%
  filter(is_train == 0)

#Change YES and No to 1 and 0
clean_labeled <- clean_labeled %>%
  left_join(tr_y, by = "id") %>%
  mutate(success = ifelse(success == "YES", 1, 0),
         success = as.factor(success))

#delete the unneeded columns
clean <- clean_labeled %>%
  select(-c(id, is_train, big_hit, backers_count))

#turn data into matrix
clean_x <- model.matrix(success~., clean)
clean_y <- clean$success

#Get numeric y for xgboost
clean_y_num <- as.numeric(clean_y)-1

#===============================

#accuracy function
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

#=============Model=============
#Define all model into function

#Tree
model_tree <- function(train, valid, minsplit, cp){
  
  #Tree control parameter
  mycontrol = rpart.control(minsplit=minsplit, cp=cp)
  
  #Full tree
  full_tree <- rpart(success ~ .,
                     method="class",
                     data=train,
                     control=mycontrol)
  
  #Pruned tree
  model<- prune(full_tree, cp=full_tree$cptable[which.min(full_tree$cptable[,"xerror"]),"CP"])
  
  #Make prediction and get the Y=1 probability predictions
  preds <- predict(model,newdata=valid)[,2]
  
  #Classification
  preds_class <- ifelse(preds > 0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class,valid$success)
  
  return(acc)
}

#Logistic
model_log <- function(train, valid){
  
  #Model
  model <- glm(success~.,
               data = train,
               family = "binomial")
  
  #Make prediction
  preds <- predict(model, newdata = valid, type = "response")
  
  #Classification
  preds_class <- ifelse(preds > 0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class,valid$success)
  
  return(acc)
}

#Ridge & Lasso
model_ridge_lasso <- function(x_train, x_valid, y_train, y_valid, alpha, grid, k){
  
  #Model
  model <- cv.glmnet(x_train,
                     y_train,
                     family = "binomial",
                     alpha = alpha,
                     lambda = grid,
                     nfolds = k)
  
  #Get the best lambda
  bestlam <- model$lambda.min
  
  #Make prediction
  preds <- predict(model, s = bestlam, newx = x_valid, type="response")
  
  #Classification
  preds_class <- ifelse(preds > 0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class, y_valid)
  
  return(acc)
}

#Random Forest - ranger
model_rf <- function(x_train, x_valid, y_train, y_valid, mtry, ntree){

  #Model
  model <- ranger(x = x_train,
                  y = y_train,
                  mtry=mtry,
                  num.trees=ntree,
                  importance="impurity",
                  probability = TRUE,
                  verbose=0)
  
  #Make prediction and get the Y=1 probability predictions
  preds <- predict(model, data=x_valid)$predictions[,2]
  
  #Classification
  preds_class <- ifelse(preds>0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class,y_valid)
  
  return(acc)
}

#Boosting - xgboost
model_xgboost <- function(x_train, x_valid, y_train, y_valid, nrounds){
  
  #Model
  model <- xgboost(data = x_train, 
                 label = y_train, 
                 max.depth = 2, 
                 eta = 1, 
                 nrounds = nrounds,  
                 objective = "binary:logistic", 
                 verbose = 0)
  
  #Make prediction
  preds <- predict(model, x_valid)
  
  #Classification
  preds_class <- ifelse(preds>0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class,y_valid)
  
  return(acc)
}

#Boosting
model_boost <- function(train, valid, n.trees, depth){
  
  #Model
  model <- gbm(success~.,
                   data=train,
                   distribution="bernoulli",
                   n.trees=n.trees,
                   interaction.depth=depth)
  #Make prediction
  preds <- predict(model, newdata=valid, type='response', n.trees=n.trees)
  
  #Classification
  preds_class <- ifelse(preds > 0.5, 1, 0)
  
  #Calculate accuracy
  acc <- accuracy(preds_class,valid$success)
  
  return(acc)
}



#===============================

#===============Feature Engineering Comparison============
#Load data
noexternal_notextmining <- read_csv("df_noexternal_notextmining.csv")
onlyexternal <- read_csv("df_onlyexternal.csv")
onlytextmining <- read_csv("df_onlytextmining.csv")

#Simple partition
tr_inds <- sample(nrow(noexternal_notextmining),.7*nrow(noexternal_notextmining))

train_noexternal_notextmining <- noexternal_notextmining[tr_inds,]
valid_noexternal_notextmining <- noexternal_notextmining[-tr_inds,]

train_onlyexternal <- onlyexternal[tr_inds,]
valid_onlyexternal <- onlyexternal[-tr_inds,]

train_onlytextmining <- onlytextmining[tr_inds,]
valid_onlytextmining <- onlytextmining[-tr_inds,]

#Calculate three accuracy under three different circumstances/feature engineering selections
acc_noexternal_notextmining <- model_log(train_noexternal_notextmining, valid_noexternal_notextmining)
acc_onlyexternal <- model_log(train_onlyexternal, valid_onlyexternal)
acc_onlytextmining <- model_log(train_onlytextmining, valid_onlytextmining)

var_selection <- c("No Variable Created from External Data and Text Mining", "Only has Variables Created from External Data", "Only has Variables Created from Text Mining")
result_acc <- c(acc_noexternal_notextmining, acc_onlyexternal, acc_onlytextmining)

compare_result <- data.frame(var_selection, result_acc)

#Save result
write.csv(compare_result, "compare_result.csv", row.names = FALSE)

#Load result
compare_result <- read_csv("compare_result.csv")
print(compare_result)
#===============================


#=============Cross-Validation=============

#Define the number of folds
k <- 5

#Separate data into k equally-sized folds
folds <- cut(seq(1,nrow(dummy_labeled_data)),breaks=k,labels=FALSE)

#define accuracy list
acc_tree_list = rep(0, k)
acc_log_list = rep(0, k)
acc_rf_list = rep(0, k)
acc_xgboost_list = rep(0, k)
acc_boost_list = rep(0, k)

#loop through each fold to calculate cross-validation accuracy
for(i in 1:k){
  
  #Segment data by fold using which() function 
  tr_inds <- which(folds==i,arr.ind=TRUE)
  
  #split data in to training and testing set
  #data for tree, logistic, and boosting
  train <- dummy_labeled_data[-tr_inds,]
  valid <- dummy_labeled_data[tr_inds,]
  
  #data for ridge, lasso, random forest, and xgboost
  x_train <- clean_x[-tr_inds,]
  x_valid <- clean_x[tr_inds,]
  
  y_train <- clean_y[-tr_inds]
  y_valid <- clean_y[tr_inds]
  
  y_train_num <- clean_y_num[-tr_inds]
  y_valid_num <- clean_y_num[tr_inds]
  
  #model_tree(training set, validation set, minsplit, cp)
  acc_tree_list[i] <- model_tree(train, valid, minsplit = 30, cp = 0.0001)
  
  #model_log(training set, validation set)
  acc_log_list[i] <- model_log(train, valid)
  
  #model_rf(training set_x, validation set_x, training set_y, validation set_y, mtry, ntree)
  acc_rf_list[i] <- model_rf(x_train, x_valid, y_train, y_valid, mtry = 10, ntree = 200)
  
  #model_xgboost(training set_x, validation set_x, training set_y, validation set_y, )
  acc_xgboost_list[i] <- model_xgboost(x_train, x_valid, y_train_num, y_valid_num, nrounds = 160)
  
  #model_boost(training set, validation set, n.tree, interaction.depth)
  acc_boost_list[i] <- model_boost(train, valid, n.trees = 100, depth=1)
}

#Calculate the mean accuracy for each model
acc_tree = mean(acc_tree_list)
acc_log = mean(acc_log_list)
acc_rf = mean(acc_rf_list)
acc_xgboost = mean(acc_xgboost_list)
acc_boost = mean(acc_boost_list)

#Since ridge and lasso have cross-validation for their function, 
#we will use the last fold as training and testing set for these two models.

#model_ridge_lasso(training set_x, validation set_x, training set_y, validation set_y, alpha, grid range, nfolds)
acc_ridge <- model_ridge_lasso(x_train, x_valid, y_train, y_valid, alpha = 0, grid = 10^seq(2,-2,length=100), k = k)
acc_lasso <- model_ridge_lasso(x_train, x_valid, y_train, y_valid, alpha = 1, grid = 10^seq(2,-2,length=100), k = k)

#Compare generalize performance
Model <- c('Tree', 'Logistic', 'Ridge', 'Lasso', 'Random Forest', 'Xgboost', 'Boosting')
Accuracy <- c(acc_tree, acc_log, acc_ridge, acc_lasso, acc_rf, acc_xgboost, acc_boost)
cv_result <- data.frame(Model, Accuracy)

#Save result
write.csv(cv_result, "cv_result.csv", row.names = FALSE)

#Load result
cv_result <- read_csv("cv_result.csv")
print(cv_result)

#==========================================

#===================Grid Search for Random Forest=============

#Simple partition
tr_inds <-  sample(nrow(dummy_labeled_data),.7*nrow(dummy_labeled_data))

x_train <- clean_x[tr_inds,]
x_valid <- clean_x[-tr_inds,]

y_train <- clean_y[tr_inds]
y_valid <- clean_y[-tr_inds]

#Define grid for mtry and ntree
arr_mtry <- c(5,10,20,50,60,75,100,200)
arr_ntree <- c(50,100,200,400,500,600,800,1000)

#Calculate the grid search size
grid_search_size <- length(arr_mtry)*length(arr_ntree)

#Define the list to store grid search paremeters and result accuracy
grid_search_rf <- array(0, dim=c(grid_search_size,3))

#For loop index
index = 1

#Grid search
for(i in 1:length(arr_mtry)){
  for(j in 1:length(arr_ntree)){

      #Get the value for mtry and ntree
      n_mtry = arr_mtry[i]
      n_ntree = arr_ntree[j]

      #Save parameters and result accuracy
      grid_search_rf[index,1] = n_mtry
      grid_search_rf[index,2] = n_ntree
      grid_search_rf[index,3] = model_rf(x_train, 
                                         x_valid, 
                                         y_train, 
                                         y_valid, 
                                         mtry = n_mtry, 
                                         ntree = n_ntree)
      
      index = index + 1
  }
}

#Save result
write.csv(grid_search_rf, "grid_search_rf_result.csv", row.names = FALSE)


#Load grid search result & plot fitting curve
grid_search_rf <- read_csv("grid_search_rf_result.csv")

#=================================================

#===================Grid Search for Boosting=============

#Simple partition
train <- dummy_labeled_data[tr_inds,]
valid <- dummy_labeled_data[-tr_inds,]

#Define grid for mtry and ntree
arr_n.tree = c(100,250,500,1000,1100,1200,1250,1300,1400,1500,1750,2000)
arr_depth = c(2,3,4,5)

#Define the list to store grid search paremeters and result accuracy
grid_search_boosting <- data.frame(matrix(ncol = 3, nrow = 0))

#Grid search
for (i in arr_n.tree) {
  for (j in arr_depth) {
    grid_search_boosting[nrow(grid_search_boosting) + 1,] = c(i, j, model_boost(train, valid, i, j))
  }
}

#Save result
write.csv(grid_search_boosting, "grid_search_boosting_result.csv", row.names = FALSE)


#Load grid search result & plot fitting curve
grid_search_boosting <- read_csv("grid_search_boosting_result.csv")

#=================================================

#================Grid Search Result Plot===============

df_boosting = read.csv("grid_search_boosting_result.csv")

df_boosting$depth <- as.character(df_boosting$depth)

ggplot(df_boosting, aes(tree_size, acc, color = depth)) + 
  geom_point() +
  geom_line() +
  labs(title="Fitting Curve", subtitle="Boosting", x="Tree Size", y="Accuracy", color="Depth")

df_rf = read.csv("grid_search_rf_result.csv")

df_rf$mtry <- as.character(df_rf$mtry)

ggplot(df_rf, aes(ntree, accuracy, color = mtry)) + 
  geom_point() +
  geom_line() +
  labs(title="Fitting Curve", subtitle="Random Forest", x="nTree", y="Accuracy", color="mtry")

#======================================================

#============Ensemble Method===============

#Model between evals pred
eval_1 = read.csv("eval_2.csv")
eval_2 = read.csv("eval_3.csv")
df_X = dummy_unlabeled_data[which(eval_1==eval_2),]
df_t = dummy_unlabeled_data[which(eval_1!=eval_2),]
df_X['success'] = if_else(eval_1[which(eval_1==eval_2),]=="YES", 1, 0)
tr_new = rbind(dummy_labeled_data, df_X)

boost.mod <- gbm(success~.,
                 data=tr_new,
                 distribution="bernoulli",
                 n.trees=1200,
                 interaction.depth=3)
#Valid train
boost_preds <- predict(boost.mod,newdata=valid,type='response',n.trees=1200)
boost_class <- ifelse(boost_preds>.5,1,0)
boost_class <- ifelse(is.na(boost_class), 0, boost_class)
accuracy(boost_class, valid$success)

#Trian all
boost_preds <- predict(boost.mod,newdata=df_t,type='response',n.trees=1200)
boost_class <- ifelse(boost_preds>.5,1,0)
boost_class <- ifelse(is.na(boost_class), 0, boost_class)

boost_class <- if_else(boost_class==1, "YES", "NO")
eval_1[which(eval_1!=eval_2),] = boost_class

#Make prediction for evaluation
write.table(eval_1, "success_group28.csv", row.names = FALSE)

#==========================================


