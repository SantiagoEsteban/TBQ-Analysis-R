# Import data
#A gentle introduction to text mining using R
library(readxl)
library(tm)
library(RTextTools)
library(NLP)
library(kernlab)
library(quanteda)
library(dplyr)
library(tidyr)
library(RWeka)
library(caret)
library(corrplot)
#library(doSNOW)
#cl<-makeCluster(3) #change the 4 to your number of CPU cores
#registerDoSNOW(cl)
#library(doParallel)

##################
#Import manual dataset created with REGEX SAS
##################

manual_grams <- as.data.frame(read_excel('tbqcompleto2015_FINAL_nodups2.xlsx'))
manual_grams <- select(manual_grams, -notbqdata, -total, -FECHA, -TEXTO)

#Analyzing frequencies
freq_manual_grams <- colSums(manual_grams)
nozero_freq_manual_grams <- freq_manual_grams > 0
sum(nozero_freq_manual_grams)
nozero_manual_grams <- manual_grams[, nozero_freq_manual_grams]

#Near zero var
nearZeroVar(nozero_manual_grams)

#Correlations
manual_grams_cor <- cor(select(nozero_manual_grams, -TBQ, -ID_PACIENTE))
corrplot(manual_grams_cor, order = "hclust", tl.cex=0.5)
#Finding highly correlated variables
findCorrelation(manual_grams_cor, cutoff = .75) #no correlations above 0.75

###############
#RANDOM FORESTS
###############

#Conditional (on screening algorithm) Manual_grams with three outcomes

nozero_manual_grams$TBQ <- make.names(nozero_manual_grams$TBQ)

train <- createDataPartition(nozero_manual_grams$TBQ,
                                            p=.8,
                                            list=F,
                                            times=1)
nozero_manual_grams_train <- nozero_manual_grams[train,]
nozero_manual_grams_test <- nozero_manual_grams[-train,]

#Varaible selection thorugh variable importance
rf.trctrl.cv <- trainControl(method='cv', number=10, classProbs = T, summaryFunction = multiClassSummary,
                                       allowParallel = F, verboseIter=T)

mtry_grid <- expand.grid(mtry=c(sqrt(142), sqrt(142)*3, sqrt(142)*4, sqrt(142)*5))

rf.nozero_manual_grams <- train(data=nozero_manual_grams_train,
                                             TBQ~. -ID_PACIENTE,
                                             method='parRF',
                                             trControl=rf.trctrl.cv,
                                             tuneGrid=mtry_grid,
                                             metric='Accuracy',
                                             maximize=T,
                                             importance=T,
                                             ntree=500,
                                            verbose=T
)
rf.varimp <- varImp(rf.nozero_manual_grams)
rf.test_results <- predict(rf.nozero_manual_grams, nozero_manual_grams_test)
confusionMatrix(rf.test_results , nozero_manual_grams_test$TBQ)

#without notbqvar
mtry_grid2 <- expand.grid(mtry=sqrt(142)*5)


rf.nozero_manual_grams2 <- train(data=nozero_manual_grams_train,
                                TBQ~. -ID_PACIENTE -notbqvar,
                                method='parRF',
                                trControl=rf.trctrl.cv,
                                tuneGrid=mtry_grid2,
                                metric='Accuracy',
                                maximize=T,
                                importance=T,
                                ntree=500,
                                verbose=T
)
rf.test_results2 <- predict(rf.nozero_manual_grams2, nozero_manual_grams_test)
confusionMatrix(rf.test_results2 , nozero_manual_grams_test$TBQ)

rf.varimp2 <- varImp(rf.nozero_manual_grams2)
rf.varimp2_df <- as.data.frame(rf.varimp2[['importance']])
morethan5imp.rf <- rf.varimp2_df[which(rf.varimp2_df$'X0' >=5
                                       & rf.varimp2_df$'X1'>=5
                                        & rf.varimp2_df$'X9'>=5),]
morethan5imp.rf <- as.vector(rownames(morethan5imp.rf))
morethan5varimp_nozero_manual_grams_train <- nozero_manual_grams_train[,colnames(nozero_manual_grams_train)%in%morethan5imp.rf]
morethan5varimp_nozero_manual_grams_train <- cbind(TBQ=nozero_manual_grams_train$TBQ, ID_PACIENTE=nozero_manual_grams_train$ID_PACIENTE, morethan5varimp_nozero_manual_grams_train)

#With varimp >5

mtry_grid3 <- expand.grid(mtry=c(sqrt(47), sqrt(47)*3, sqrt(47)*5, sqrt(47)*6))


rf.trctrl.cv2 <- trainControl(method='cv', number=5, classProbs = T, summaryFunction = multiClassSummary,
                             allowParallel = F, verboseIter=T)

rf.nozero_manual_grams3 <- train(TBQ~. -ID_PACIENTE, 
                                 data=morethan5varimp_nozero_manual_grams_train,
                                 method='parRF',
                                 trControl=rf.trctrl.cv2,
                                 tuneGrid=mtry_grid3,
                                 metric='Accuracy',
                                 maximize=T,
                                 importance=T,
                                 ntree=500,
                                 verbose=T
)
rf.test_results3 <- predict(rf.nozero_manual_grams3, nozero_manual_grams_test)
confusionMatrix(rf.test_results , nozero_manual_grams_test$TBQ)

