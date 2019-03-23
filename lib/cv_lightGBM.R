######################################
### Cross Validation for light GBM ###
######################################

cv.lgb <- function(X.train, y.train, d, nr, K){
  
  n <- dim(y.train)[1]
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.lgb.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i, ,]
    train.label <- y.train[s != i, ,]
    test.data <- X.train[s == i, ,]
    test.label <- y.train[s == i, ,]
    
    par <- list(depth=d, nr = nr)
    fit <- train.lgb(train.data, train.label, par)
    pred <- test.lgb(fit, test.data)  
    cv.lgb.error[i] <- mean((pred - test.label)^2)  
    
  }			
  return(c(mean(cv.lgb.error),sd(cv.lgb.error)))
}


nrseq <- seq(200, 200, 10)
err_cv_lgb <- cbind(nrseq, array(dim=c(length(nrseq), 2)))
colnames(err_cv_lgb) <- c("Num_iterations", "mean_cv_error", "sd_cv_error")
for (i in 1:length(nrseq)){
  err_cv_lgb[i,] <- cv.lgb(dat_train$feature, dat_train$label, nr = nrseq[i], d = 50, K = 5)
}

save(err_cv_lgb, file="../output/err_cv_lgb.RData")