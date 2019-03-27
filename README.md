# Project: Can you unscramble a blurry image? 
![image](figs/example.png)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2019

+ Team 8
+ Team members
	+ Chen, Xishi xc2455@columbia.edu
	+ Ren, Xueqi xr2134@columbia.edu
	+ Vitha, Matthew mv2705@columbia.edu
	+ Wang, Yujie yw3285@columbia.edu
	+ Zhao, Xiaoxi xz2740@columbia.edu

+ Project summary: In this project, we created a classification engine for enhance the resolution of images. For the baseline model, we first pad the image (add 0-value pixels to the edges) and extract the 8 points around the points we want to predict, calculate the difference between these 8 points and the center point and use these 8 differences to predict the 4 differences value in RGB in the high resolution image. We then add the center point RGB to the 4 differences to get the predicted values of the 4 pixels in high resolution image. The model used to predict is gradient boosting model with max depth of 7. When we applied the Xgboost model, we also noticed that it finished calculation in 20 minutes, while GBM model need one day and a half. We choose eta=0.5,depth=7, nrounds=11. 
We also tried light GBM, Neural Network and SRGAN as well. But due to the running time for superresolution and the psnr for each method, we finally choose xgboost as the final improved model.

	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

+ Chen, Xishi: complete the Xgboost model including finishing mainXGboost.rmd, cross_validationXGboost.r, superResolutionXG.r, trainXGboost.r and testXGboost.r; Try different parameters to improve the performance of the model.

+ Ren, Xueqi: complete the Multi-Layer Perceptron model (basic neural network) including finihsing mainNN.Rmd, superResolutionNN.R, trainNN.R and testNN.R. 

+ Vitha, Matthew: attempted to train a GAN NN for predictions but came up with complications. Code for the network setup is in Network-gan.py, while the train file is in train_gan.py

+ Wang, Yujie: trained, tested, performed cross validation for parameter tuning, produced super resolution images using light GBM model. 

+ Zhao, Xiaoxi: complete the baseline model with finishing feature.R, superresolution.R and change the main.R to do the train-validation set splitting; tune the baseline model using cross validation; Improve the XGBoost model by parameter tuning:eta, max_depth and nrounds.Write the final code for superrosolution.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
