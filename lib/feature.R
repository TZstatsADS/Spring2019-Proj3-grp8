#############################################################
### Construct features and responses for training images###
#############################################################

### Authors: Chengliang Tang/Tian Zheng
### Project 3

feature <- function(LR_dir, HR_dir, n_points=10,train_index){
  
  ### Construct process features for training images (LR/HR pairs)
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + number of points sampled from each LR image
  ### Output: an .RData file contains processed features and responses for the images
  
  ### load libraries
  library("EBImage")
  library(plyr)
  n_files<-length(train_index)
  ### store feature and responses
  featMat <- array(NA, c(n_files * n_points, 8, 3))
  labMat <- array(NA, c(n_files * n_points, 4, 3))
  get_feature_lr<-function(index,pad){
    x<-index[1]+1
    y<-index[2]+1
    return(c(pad[x-1,y-1],pad[x,y-1],pad[x+1,y-1],pad[x-1,y],pad[x+1,y],pad[x-1,y-1],pad[x,y-1],pad[x+1,y-1]))
  }
  get_feature_hr<-function(index,pad){
    x<-index[1]
    y<-index[2]
    return(c(pad[2*x-1,2*y-1],pad[2*x,2*y-1],pad[2*x-1,2*y],pad[x*2,y*2]))
  }
  ### read LR/HR image pairs
  for(k in 1:n_files){
    i=train_index[k]
    imgLR <- readImage(paste0(LR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgHR <- readImage(paste0(HR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    ### step 1. sample n_points from imgLR
    set.seed(200)
    rownum<- dim(imgLR)[2]
    colnum<- dim(imgLR)[1]
    x<-sample(1:colnum,n_points,replace=T)
    y<-sample(1:rownum,n_points,replace=T)
    index<-cbind(x,y)
    ### step 2. for each sampled point in imgLR,
    
        ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat
        ###           tips: padding zeros for boundary points
    for (j in 1:3){
      p<-rbind(rep(0,rownum),as.matrix(imgLR[ ,,j]),rep(0,rownum))
      pad2<-cbind(rep(0,colnum+2),p,rep(0,colnum+2))
      pad3<-as.matrix(imgHR[,,j])
      featurelr<-aaply(index,1,get_feature_lr,pad2)
      featurehr<-aaply(index,1,get_feature_hr,pad3)
      featMat[((k-1)*n_points+1):(n_points*k),,j]<-featurelr
      labMat[((k-1)*n_points+1):(n_points*k),,j]<-featurehr
      
    }

    
    
        ### step 2.2. save the corresponding 4 sub-pixels of imgHR in labMat
    
    ### step 3. repeat above for three channels
      
  }
  return(list(feature = featMat, label = labMat))
}
