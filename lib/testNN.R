######################################################
### Fit the regression model with testing data ###
######################################################

### Author: Chengliang Tang
### Project 3

testNN <- function(modelList, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model list using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(keras)
  
  predArr <- array(NA, c(dim(dat_test)[1], 4, 3))
  
  for (i in 1:12){
    print(i)
    model <- modelList[[i]]
    ### calculate column and channel
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_test[, , c2]
    
    ### make predictions
    #model <- load_model_hdf5(paste(paste("../output/NNmodels/NNmodel", toString(i), sep=""), ".h5", sep=""))
    predArr[, c1, c2] <- model %>% predict(featMat)
  }
  
  return(as.numeric(predArr))
}