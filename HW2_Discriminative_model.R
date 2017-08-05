# -- Machine Learning 2017 -- 
# HW 2 : Linear models for classification
# method : Probabilistic Discriminative model
# Multi-class logistic regression 
# Date : 2017/3/23

# [0] read in data ------------------------------------------------------------
path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/HW2/") # Windows
path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/HW2/") # Mac

library(bmp)
library(magrittr)
library(dplyr)
library(ggplot2)

# This function can get picture's pixels, and save as dataframe
get_Training_Matrix<- function(class,num){
  M <- matrix(0,1,900)
  for(i in 1:num){
    tmp <- read.bmp(paste0(path,"/Data_Train/Class",class,"/faceTrain",class,"_",i,".bmp")) %>% 
      t %>% as.vector() 
    M <- rbind(M,tmp)
  }
  M <- as.data.frame(M)
  M <- data.frame(class = rep(paste0("C",class),1000),
                  num = 1:num) %>% cbind.data.frame(.,M[-1,])
  return(M)
}

# get row data
train_raw <- rbind.data.frame(get_Training_Matrix(1,1000),
                              get_Training_Matrix(2,1000),
                              get_Training_Matrix(3,1000))

# seperate data 
# balance data ( train:valid = 9:1 )
  D_train <- train_raw %>% filter((num %% 10) != 0)
  D_valid <- train_raw %>% filter((num %% 10) == 0)

# unbalance data  
  D_train <- train_raw %>% filter(class == "C1" & (num %% 10) == 2 |
                                  class == "C2" & (num %% 10) != 2 |
                                  class == "C3" & (num %% 10) != 2)

  D_valid <- train_raw %>% filter(class == "C1" & (num %% 10) != 2 |
                                 class == "C2" & (num %% 10) == 2 |
                                  class == "C3" & (num %% 10) == 2)

# [1] Principle Component Analysis (PCA) -------------------------------------------------------

# This function can get Covariance matrix's eigenvector
  get_PCA_eigenvector <- function(dt,n){
    dt <- as.matrix(dt)
    dt <- scale(dt, center = TRUE, scale = TRUE) # normalized
    dt.cov <- cov(dt) # covariance
    dt.eigen <- eigen(dt.cov) # eigenvectors & eigenvalues
    return(dt.eigen$vectors[ ,1:n])
  }

# Given 'eigenvector' 'mean vector' 'sd vector'
# and this function will give you PCA data ( 2-dim, PC1 & PC2 )
  get_PCA_data <- function(dt,eigenvector,mean_vector,sd_vector){
    dt <- as.matrix(dt)
    dt <- dt - rep(mean_vector,each=nrow(dt)) 
    dt <- dt / rep(sd_vector,each=nrow(dt))
    PCA_data <- dt %*% eigenvector
    colnames(PCA_data) <- c("PC1", "PC2")
    return(PCA_data)
  }

# get PCA parameters from training data
  mean_vector <- apply(D_train[,c(3:902)],2,mean)
  sd_vector <- apply(D_train[,c(3:902)],2,sd)
  train_eigenvector <- get_PCA_eigenvector(D_train[,c(3:902)], n=2)

# save PCA result (for training data)
  train_PCA_data <- get_PCA_data(D_train[,c(3:902)],train_eigenvector,mean_vector,sd_vector)
  D_train_PCA <- cbind.data.frame(D_train[,1:2],as.data.frame(train_PCA_data))

# [2] Model construction (Probabilistic Discriminate Model) ------------------------------
  
  # phi 
  phi <- D_train_PCA[,c(3,4)] %>% as.matrix()
  
  # creat block matrix (for compute easier)
  library(magic) #creat block matrix
  big_phi <- adiag(phi,phi,phi)

  # initial weight vector setting
  # 2 X 3
  weight_vectors <- matrix(c(1, 1, 1, 1, 1, 1),2,3)

  # True value matrix
  # 2700 X 3
  T_matrix <- rbind(matrix(rep(c(1,0,0),each=900),900,3),
                    matrix(rep(c(0,1,0),each=900),900,3),
                    matrix(rep(c(0,0,1),each=900),900,3))
  
  # Y matrix [Y1 Y2 Y3]
  # 2700 X 3
   Y <- exp(phi %*% weight_vectors) / (exp(phi %*% weight_vectors)[,1] +
                                       exp(phi %*% weight_vectors)[,2] +
                                       exp(phi %*% weight_vectors)[,3] )
   
  # gradient E(w)
  # 6 X 1
  gradient_Ew <- t(big_phi) %*% ((Y-T_matrix) %>% as.vector())
  
  # get  y_nk * (I_kj - Y_nj)
  get_submatrix <- function(Y,k,j){
    R <- matrix(0,nrow(Y),nrow(Y))
    if (k == j){
      for(i in 1:nrow(Y)){R[i,i] <- Y[i,k]*(1-Y[i,k])}}
    
    else {
      for(i in 1:nrow(Y)){R[i,i] <- Y[i,k]*(0-Y[i,j])}}
    
    return(R) 
  }
  
  big_R <- rbind( cbind(get_submatrix(Y,1,1),get_submatrix(Y,1,2),get_submatrix(Y,1,3)),
                  cbind(get_submatrix(Y,2,1),get_submatrix(Y,2,2),get_submatrix(Y,2,3)),
                  cbind(get_submatrix(Y,3,1),get_submatrix(Y,3,2),get_submatrix(Y,3,3)))
  
  # Hessian matrix
  H <- t(big_phi) %*% big_R %*% big_phi
  # det(H)
  # solve(H)
  # solve(H, tol=1e-18) %*% H
  # H %*% solve(H, tol=1e-18) 
  
  # Newton Raphson iteration
  # weight_vectors <- weight_vectors + matrix(solve(H, tol=1e-18) %*% gradient_Ew,2,3)
  weight_vectors <- weight_vectors - matrix(solve(H, tol=1e-18) %*% gradient_Ew,2,3)
  
  weight_vectors
  