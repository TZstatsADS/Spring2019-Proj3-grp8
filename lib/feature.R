#############################################################
### Construct features and responses for training images###
#############################################################

### Authors: Chengliang Tang/Tian Zheng
### Project 3

feature <- function(LR_dir, HR_dir, n_points=1000){
  
  ### Construct process features for training images (LR/HR pairs)
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + number of points sampled from each LR image
  ### Output: an .RData file contains processed features and responses for the images
  
  ### load libraries
  library("EBImage")
  n_files <- length(list.files(LR_dir))
  
  #print(n_files)
  
  get_features_LR<- function(lr_x, lr_y){
    output<- array(0,c(8, 3))
    Around<- PixelM[c(lr_y-1,lr_y,lr_y+1), c(lr_x-1, lr_x, lr_x+1), ]
    Scaled<- Around - rep(Around[2,2, ], each=9) 
    output[1:3, ]<- Scaled[1, , ]
    output[4:5, ]<- Scaled[2, c(1, 3), ]
    output[6:8, ]<- Scaled[3, , ]
    return(output)
  }
  get_Labels_HR<- function(hr_x, hr_y){
    output<- array(0, c(4,3))
    Target<- Pixel_H[c(hr_y-1, hr_y), c(hr_x-1, hr_x), ]
    Scaled<- Target - rep(PixelM[hr_y/2+1, hr_x/2+1, ], each=4)
    output[1:2, ]<- Scaled[1, , ]
    output[3:4, ]<- Scaled[2, , ]
    return(output)
  }
  
  
  ### store feature and responses
  featMat <- array(NA, c(n_files * n_points, 8, 3))
  labMat <- array(NA, c(n_files * n_points, 4, 3))
  
  ### read LR/HR image pairs
  for(i in 1:n_files){
    imgLR <- readImage(paste0(LR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgHR <- readImage(paste0(HR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    ### step 1. sample n_points from imgLR
    lrxd<- dim(imgLR)[2]
    lryd<- dim(imgLR)[1]
    
    x<- sample(lrxd, n_points, replace=TRUE)
    y<- sample(lryd, n_points, replace=TRUE)
    

    
    ### step 2. for each sampled point in imgLR,
    
    ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat
    ###           tips: padding zeros for boundary points
    
    M1<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 1]), rep(0, lryd)), rep(0, lrxd+2))
    M2<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 2]), rep(0, lryd)), rep(0, lrxd+2))
    M3<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 3]), rep(0, lryd)), rep(0, lrxd+2))
    PixelM<- array(0, c(lryd+2, lrxd+2, 3))
    PixelM[ , , 1]<- M1
    PixelM[ , , 2]<- M2
    PixelM[ , , 3]<- M3
    
    left<- 1000*(i-1)+1
    right<- 1000*i
    Features<- mapply(get_features_LR, x+1, y+1)
    featMat[left:right, , 1]<- t(Features[1:8, ])
    featMat[left:right, , 2]<- t(Features[9:16, ])
    featMat[left:right, , 3]<- t(Features[17:24, ])
    
    ### step 2.2. save the corresponding 4 sub-pixels of imgHR in labMat
    Pixel_H<- as.array(imgHR)
    
    Features<- mapply(get_Labels_HR, 2*x, 2*y)
    labMat[left:right, , 1]<- t(Features[1:4, ])
    labMat[left:right, , 2]<- t(Features[5:8, ])
    labMat[left:right, , 3]<- t(Features[9:12, ])
    
    ### step 3. repeat above for three channels
    
  }
  return(list(feature = featMat, label = labMat))
}

