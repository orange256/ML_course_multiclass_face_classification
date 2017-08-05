# -- Machine Learning 2017 -- 
# HW 2 : Linear models for classification
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
  
  
# [2] Model construction (Probabilistic Generative Model) ------------------------------
  
# p(C_k)
  p_C1 <- table(D_train_PCA$class)[1] / length(D_train_PCA$class)
  p_C2 <- table(D_train_PCA$class)[2] / length(D_train_PCA$class)
  p_C3 <- table(D_train_PCA$class)[3] / length(D_train_PCA$class)

# compute u_k & sigma_k  
  u_k <- aggregate(D_train_PCA[, 3:4], list(D_train_PCA$class), mean)
  u_1 <- u_k[1,2:3] %>% as.matrix()
  u_2 <- u_k[2,2:3] %>% as.matrix()
  u_3 <- u_k[3,2:3] %>% as.matrix()
  
  sigma_1 <- cov(subset(D_train_PCA, class == "C1", select = c(PC1,PC2)))
  sigma_2 <- cov(subset(D_train_PCA, class == "C2", select = c(PC1,PC2)))
  sigma_3 <- cov(subset(D_train_PCA, class == "C3", select = c(PC1,PC2))) 
  
# likelihood function : p(x|C_k)
  p_x_given_C1 <- function(x){
    x <- as.matrix(x)
    tmp <- (2*pi)**(-3/2) * det(sigma_1)**(-1/2) * exp((-1/2) * (x - u_1) %*% solve(sigma_1) %*% t(x - u_1))
    return(tmp)
  }
  
  p_x_given_C2 <- function(x){
    x <- as.matrix(x)
    tmp <- (2*pi)**(-3/2) * det(sigma_2)**(-1/2) * exp((-1/2) * (x - u_2) %*% solve(sigma_2) %*% t(x - u_2))
    return(tmp)
  }  
  
  p_x_given_C3 <- function(x){
    x <- as.matrix(x)
    tmp <- (2*pi)**(-3/2) * det(sigma_3)**(-1/2) * exp((-1/2) * (x - u_3) %*% solve(sigma_3) %*% t(x - u_3))
    return(tmp)
  } 
  
# p(C_k|x)
  p_C1_given_x <- function(x){
    x <- as.matrix(x)
    tmp <- p_x_given_C1(x)*p_C1 / (p_x_given_C1(x)*p_C1 + p_x_given_C2(x)*p_C2 + p_x_given_C3(x)*p_C3)
    return(tmp)
  }
  
  p_C2_given_x <- function(x){
    x <- as.matrix(x)
    tmp <- p_x_given_C2(x)*p_C2 / (p_x_given_C1(x)*p_C1 + p_x_given_C2(x)*p_C2 + p_x_given_C3(x)*p_C3)
    return(tmp)
  }
  
  p_C3_given_x <- function(x){
    x <- as.matrix(x) 
    tmp <- p_x_given_C3(x)*p_C3 / (p_x_given_C1(x)*p_C1 + p_x_given_C2(x)*p_C2 + p_x_given_C3(x)*p_C3)
    return(tmp)
  }


# [3] validation data --------------------------------------------------------------------

# save PCA result (for validation data)
  valid_PCA_data <- get_PCA_data(D_valid[,c(3:902)],train_eigenvector,mean_vector,sd_vector)  
  D_valid_PCA <- cbind.data.frame(D_valid[,1:2],as.data.frame(valid_PCA_data))
  
  
# compute p(C_k|x), k =1,2,3
  for(i in 1:nrow(D_valid_PCA)){
    D_valid_PCA$Prob_C1[i] <- p_C1_given_x(D_valid_PCA[i,3:4])
    D_valid_PCA$Prob_C2[i] <- p_C2_given_x(D_valid_PCA[i,3:4])
    D_valid_PCA$Prob_C3[i] <- p_C3_given_x(D_valid_PCA[i,3:4])
  }
  
# choose biggest one 
  for(i in 1:nrow(D_valid_PCA)){
    D_valid_PCA$which_is_max[i] <- which.max(D_valid_PCA[i,5:7])
  }
# [4] Error rate -------------------------------------------------------------
  hit_table <- table(D_valid_PCA$class, D_valid_PCA$which_is_max)
  hit_table
  
  hit_rate <- sum(diag(hit_table)) / nrow(D_valid_PCA)
  hit_rate

  error_rate <- 1 - hit_rate
  error_rate
  
# [5] PCA plot ----------------------------------------------------------------------------
# training data
  ggplot(D_train_PCA,aes(x=PC1, y=PC2, group = class) ) + 
    geom_point(aes(color = class, shape = class))+
    stat_ellipse()
  
# decision region
  # fake data
  decision <- data.frame(PC1 = runif(5000,-30,35),
                         PC2 = runif(5000,-30,20)) 
  
  # compute p(C_k|x), k =1,2,3
  for(i in 1:nrow(decision)){
    decision$Prob_C1[i] <- p_C1_given_x(decision[i,1:2])
    decision$Prob_C2[i] <- p_C2_given_x(decision[i,1:2])
    decision$Prob_C3[i] <- p_C3_given_x(decision[i,1:2])
  }
  
  # choose biggest one 
  for(i in 1:nrow(decision)){
    decision$which_is_max[i] <- which.max(decision[i,3:5])
  }
  decision$which_is_max <-as.factor(decision$which_is_max)
  
  # plot decision region
  ggplot(decision,aes(x=PC1, y=PC2, group = which_is_max) ) + 
    geom_point(aes(color = which_is_max, shape = which_is_max))
  
# validation data  
  ggplot(D_valid_PCA,aes(x=PC1, y=PC2, group = class) ) + 
    geom_point(aes(color = class, shape = class))+
    stat_ellipse()
  
# [6] testing data ------------------------------------------------------------------------------------
  # This function can get picture's pixels, and save as dataframe
  get_Testing_Matrix<- function(num){
    M <- matrix(0,1,900)
    for(i in 1:num){
      tmp <- read.bmp(paste0(path,"/Demo/",i,".bmp")) %>% t %>% as.vector() 
      M <- rbind(M,tmp)
    }
    M <- as.data.frame(M[-1,])
    return(M)
  }

  # get raw testing data 
  test_raw <- get_Testing_Matrix(600)  

  # save PCA result (for testing data)
  test_PCA_data <- get_PCA_data(test_raw,train_eigenvector,mean_vector,sd_vector)  
  D_test_PCA <- cbind(matrix(0,600,2),test_PCA_data) %>% as.data.frame()
  
  # compute p(C_k|x), k =1,2,3
  for(i in 1:nrow(D_test_PCA)){
    D_test_PCA$Prob_C1[i] <- p_C1_given_x(D_test_PCA[i,3:4])
    D_test_PCA$Prob_C2[i] <- p_C2_given_x(D_test_PCA[i,3:4])
    D_test_PCA$Prob_C3[i] <- p_C3_given_x(D_test_PCA[i,3:4])
  }
  
  # choose biggest one 
  for(i in 1:nrow(D_test_PCA)){
    D_test_PCA$which_is_max[i] <- which.max(D_test_PCA[i,5:7])
  }

  # get demo_output (600 X 3)
  get_demo_target <- function(data){
    demo_output <- matrix(0,600,3)
    for(i in 1:nrow(data)){
      if(data$which_is_max == 1){demo_output[i,] <- c(1,0,0)}
      else if(data$which_is_max == 2){demo_output[i,] <- c(0,1,0)}
      else{demo_output[i,] <- c(0,0,1)}
    }
    return(demo_output)
  }
  
  # save result as "demo_target"
  demo_target <- get_demo_target(D_test_PCA)

# [7] save output ---------------------------------------------------------------------
  write.table(demo_target,file = paste0(path,"DemoTarget.csv"),col.names = F,row.names = F,sep = ",")

  
  
# [8] other discussion ------------------------------------------------
  PCA <- prcomp(D_train[,c(3:902)], scale = TRUE)
  plot(PCA, type="line", main="Scree Plot for PCA") 
  
  # compute variance and it's %
  vars <- (PCA$sdev)^2  
  props <- vars / sum(vars)    
  
  
  # plot cumulative props
  cumulative.props <- cumsum(props)  
  plot(cumulative.props)

  # find PC1 , PC2
  top2.pca.eigenvector <- PCA$rotation[, 1:2]
  first.pca  <- top2.pca.eigenvector[, 1]  # PC1
  second.pca <- top2.pca.eigenvector[, 2]  # PC2
  
  # plot loading for PC1
  dotchart(head(first.pca[order(first.pca, decreasing=T)],30) ,
           main="Loading Plot for PC1",                      
           xlab="Variable Loadings",                        
           col="red")                                        
  
  
   tmp <- head(first.pca[order(first.pca, decreasing=T)],300) %>% as.data.frame()
   tmp <- rownames(tmp) %>% substr(start=2,stop = 4) %>% as.numeric() %>% as.data.frame()
   tmp <- tmp %>% mutate(Y=30-.%/%30, X=.%%30)
   plot(tmp$X,tmp$Y,xlim=c(1, 30),ylim=c(1, 30), main="PC1")
   
   tmp <- head(second.pca[order(second.pca, decreasing=T)],300) %>% as.data.frame()
   tmp <- rownames(tmp) %>% substr(start=2,stop = 4) %>% as.numeric() %>% as.data.frame()
   tmp <- tmp %>% mutate(Y=30-.%/%30, X=.%%30)
   plot(tmp$X,tmp$Y,xlim=c(1, 30),ylim=c(1, 30), main="PC2")
  
  # end ------------------------------------------------------------------------------------
  