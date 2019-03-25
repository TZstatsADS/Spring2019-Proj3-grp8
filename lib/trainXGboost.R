#########################################################
### Train a classification model with training features ###
#########################################################

### Author: Chengliang Tang
### Project 3


trainXG <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  features from LR images 
  ###  -  responses from HR images
  ### Output: a list for trained models
  
  ### load libraries
  if(!require(BayesTree)) install.packages("BayesTree")
  if(!require(iRF)) install.packages("iRF")
  if(!require(rpart)) install.packages("rpart")
  if(!require(xgboost)) install.packages("xgboost")
  library("BayesTree")
  library("iRF")
  library("rpart")
  library("xgboost")
  
  ### creat model list
  modelList <- list()
  
  ### Train with gradient boosting model
  if(is.null(par)){
    nr<- 10
  } else {
    nr<- par$nr
  }
  ### the dimension of response arrat is * x 4 x 3, which requires 12 classifiers
  ### this part can be parallelized
  for (i in 1:12){
    ## calculate column and channel
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_train[, , c2]
    labMat <- label_train[, c1, c2]
    fit_improved <- xgboost(data = as.matrix(featMat), 
                            label = as.matrix(labMat), 
                            booster = "gblinear", 
                            eta = 0.1,
                            nrounds = nr, 
                            max_depth=7,
                            verbose = FALSE, 
                            objective = "reg:linear", 
                            eval_metric = "rmse", 
                            lambda = 1, 
                            alpha = 0)
    modelList[[i]] <- list(fit=fit_improved, iter=par)
  }
  
  return(modelList)
}
