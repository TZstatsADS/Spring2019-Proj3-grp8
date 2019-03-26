#########################################################
### Train a classification model with training features ###
#########################################################

### Author: Chengliang Tang
### Project 3


trainNN <- function(dat_train, label_train){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  features from LR images 
  ###  -  responses from HR images
  ### Output: a list for trained models
  
  ### load libraries
  library(keras)
  
  ### creat model list
  # modelList <- list()
  
  ### the dimension of response array is * x 4 x 3, which requires 12 classifiers
  ### this part can be parallelized
  for (i in 1:12){
    ## calculate column and channel
    
    c1 <- (i-1) %% 4 + 1
    c2 <- (i-c1) %/% 4 + 1
    featMat <- dat_train[, , c2]
    labMat <- label_train[, c1, c2]

    
    # Initialize a sequential model
    model <- keras_model_sequential() 
    
    # Add layers to the model
    model %>% 
      layer_dense(units = 512, activation = "relu", input_shape = c(8)) %>% 
      layer_dense(units = 1, activation = 'tanh')
    
    # Compile the model
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = 'adam'
    )
    
    # Save the training history in history
    model %>% fit(
      featMat, labMat, 
      epochs = 5, batch_size = 1000,
      validation_split = 0.2
    )
    
    model %>% save_model_hdf5(paste(paste("../output/NNmodels/NNmodel", toString(i), sep=""), ".h5", sep=""))
    # modelList[[i]] <- model
  }
  
  # return(modelList)
  return()
}
