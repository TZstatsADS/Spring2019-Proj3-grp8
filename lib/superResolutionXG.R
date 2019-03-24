########################
### Super-resolution ###
########################

### Author: Chengliang Tang
### Project 3

superResolutionXG <- function(LR_dir, HR_dir, modelList,index){
  
  ### Construct high-resolution images from low-resolution images with trained predictor
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + a list for predictors
  
  ### load libraries
  library("EBImage")
  n_files <- length(index)
  get_features_LR<- function(lr_x, lr_y){
    output<- array(0,c(8, 3))
    Around<- PixelM[c(lr_y-1,lr_y,lr_y+1), c(lr_x-1, lr_x, lr_x+1), ]
    Scaled<- Around - rep(Around[2,2, ], each=9) 
    output[1:3, ]<- Scaled[1, , ]
    output[4:5, ]<- Scaled[2, c(1, 3), ]
    output[6:8, ]<- Scaled[3, , ]
    return(output)
  }
  mse<-c()
  ### read LR/HR image pairs
  for(k in 1:n_files){
    i=index[k]
    imgLR <- readImage(paste0(LR_dir,  "img", "_", sprintf("%04d", i), ".jpg"))
    pathHR <- paste0(HR_dir,  "img", "_", sprintf("%04d", i), ".jpg")
    featMat <- array(NA, c(dim(imgLR)[1] * dim(imgLR)[2], 8, 3))
    
    ### step 1. for each pixel and each channel in imgLR:
    ###           save (the neighbor 8 pixels - central pixel) in featMat
    ###           tips: padding zeros for boundary points
    lrxd<- dim(imgLR)[2]
    lryd<- dim(imgLR)[1]
    M1<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 1]), rep(0, lryd)), rep(0, lrxd+2))
    M2<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 2]), rep(0, lryd)), rep(0, lrxd+2))
    M3<- rbind(rep(0, lrxd+2), cbind(rep(0, lryd), as.matrix(imgLR[ , , 3]), rep(0, lryd)), rep(0, lrxd+2))
    PixelM<- array(0, c(lryd+2, lrxd+2, 3))
    PixelM[ , , 1]<- M1
    PixelM[ , , 2]<- M2
    PixelM[ , , 3]<- M3
    x<-rep(1:lrxd,each=lryd)
    y<-rep(1:lryd,lrxd)
    Features<- mapply(get_features_LR, x+1, y+1)
    featMat[, , 1]<- t(Features[1:8, ])
    featMat[, , 2]<- t(Features[9:16, ])
    featMat[, , 3]<- t(Features[17:24, ])
    ### step 2. apply the modelList over featMat
    predMat <- testXG(modelList, featMat)
    
    ### step 3. recover high-resolution from predMat and save in HR_dir
    predMat_array <- array(predMat, c(dim(featMat)[1], 4, 3))
    imgoutputhr<-array(NA,c(lryd*2,lrxd*2,3))
    
    LRcenter <- array(imgLR, c(lryd*lrxd, 1, 3))
    predMat_recov<-abind(LRcenter,LRcenter,LRcenter,LRcenter,along=2)
    predMat_recov<-predMat_array+predMat_recov
    
    for(l in 1:3){
      y13 <- as.vector(t(predMat_recov[, c(1,3),l] ))
      y24 <- as.vector(t(predMat_recov[, c(2,4),l] ))
      
      imgoutputhr[, seq(1, 2*lrxd-1, 2), l] <- y13
      imgoutputhr[, seq(2, 2*lrxd, 2), l] <- y24
    }
    
    img_out<- Image(imgoutputhr, colormode="Color")
    display(img_out)
    writeImage(img_out, paste0("../data/train/testHRXG/",  "img", "_", sprintf("%04d", i), ".jpg"))
    ### step 4. report test MSE and PSNR
    testimgHR <- readImage(paste0(HR_dir,  "img", "_", sprintf("%04d", i), ".jpg"))
    mse<-c(mse,mean((imgoutputhr - testimgHR)^2))
    
    
  }
  testmse<-mean(mse)
  testpsnr<- -10*log10(testmse)
  return(c(testmse,testpsnr))
  
}
