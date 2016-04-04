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
library(doSNOW)
cl <- makeCluster(3, type = "SOCK") #change the 4 to your number of CPU cores
registerDoSNOW(cl)

##################
#Import manual dataset created with REGEX SAS
##################

manual_grams <- as.data.frame(read_excel('tbqcompleto2015_FINAL_nodups2.xlsx'))
manual_grams2 <- select(manual_grams, -notbqdata, -total, -FECHA, -TEXTO)

#Analyzing frequencies
freq_manual_grams <- colSums(manual_grams2)
nozero_freq_manual_grams <- freq_manual_grams > 0
sum(nozero_freq_manual_grams)
nozero_manual_grams <- manual_grams2[, nozero_freq_manual_grams]

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
rf.trctrl.cv <- trainControl(method='cv', number=2, classProbs = T, summaryFunction = multiClassSummary,
                                       allowParallel = T, verboseIter=T)

mtry_grid <- expand.grid(mtry=c(sqrt(142), sqrt(142)*3, sqrt(142)*4, sqrt(142)*5))

rf.nozero_manual_grams <- train(data=nozero_manual_grams_train,
                                             TBQ~. -ID_PACIENTE,
                                             method='parRF',
                                             trControl=rf.trctrl.cv,
                                             tuneGrid=mtry_grid,
                                             metric='Accuracy',
                                             maximize=T,
                                             importance=T,
                                             ntree=100,
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

rf.varimp2 <- varImp(rf.nozero_manual_grams2, scale=F)
rf.varimp2_df <- as.data.frame(rf.varimp2[['importance']])
morethan5imp.rf <- rf.varimp2_df[which(rf.varimp2_df$'X0' > 0
                                       & rf.varimp2_df$'X1'> 0
                                        & rf.varimp2_df$'X9'> 0),]
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
plot(varImp(rf.nozero_manual_grams3))
plot(rf.nozero_manual_grams3)

############
#Boosted trees
############

boostedTree.trctrl <- trainControl(method='cv', 
                                                number=5, 
                                                classProbs = T, 
                                                summaryFunction = multiClassSummary, 
                                                verboseIter=T,
                                                allowParallel=T)


tuneGrid.gbm <- expand.grid(n.trees=c(200,300,400),
                            interaction.depth=c(5:9),
                            shrinkage=c(0.1),
                            n.minobsinnode=10)
system.time(
    boostedTree_3outcomes <- train(data=morethan5varimp_nozero_manual_grams_train,
                                   TBQ~. -ID_PACIENTE,
                                   distribution='multinomial',
                                   method='gbm',
                                   metric='Accuracy',
                                   trControl=boostedTree.trctrl,
                                   tuneGrid=tuneGrid.gbm,
                                   verbose=T)
)

plot(boostedTree_3outcomes)

tuneGrid.gbm2 <- expand.grid(n.trees=c(400,500,600),
                            interaction.depth=c(7:9),
                            shrinkage=c(0.1),
                            n.minobsinnode=10)

system.time(
    boostedTree_3outcomes <- train(data=morethan5varimp_nozero_manual_grams_train,
                                   TBQ~. -ID_PACIENTE,
                                   distribution='multinomial',
                                   method='gbm',
                                   metric='Accuracy',
                                   trControl=boostedTree.trctrl,
                                   tuneGrid=tuneGrid.gbm2,
                                   verbose=T)
)
#Acc0.93

tuneGrid.gbm3 <- expand.grid(n.trees=c(1000),
                             interaction.depth=c(8),
                             shrinkage=c(0.1),
                             n.minobsinnode=10)

system.time(
    boostedTree_3outcomes2 <- train(data=morethan5varimp_nozero_manual_grams_train,
                                   TBQ~. -ID_PACIENTE,
                                   distribution='multinomial',
                                   method='gbm',
                                   metric='Accuracy',
                                   trControl=boostedTree.trctrl,
                                   tuneGrid=tuneGrid.gbm3,
                                   verbose=T)

#Complete dataset
tuneGrid.gbm4 <- expand.grid(n.trees=c(300),
                             interaction.depth=c(8),
                             shrinkage=c(0.1),
                             n.minobsinnode=10)

boostedTree.trctrl2 <- trainControl(method='cv', 
                                   number=2, 
                                   classProbs = T, 
                                   summaryFunction = multiClassSummary, 
                                   verboseIter=T,
                                   allowParallel=T)

system.time(
    boostedTree_3outcomes3 <- train(data=nozero_manual_grams_train,
                                    TBQ~. -ID_PACIENTE -notbqvar,
                                    distribution='multinomial',
                                    method='gbm',
                                    metric='Accuracy',
                                    trControl=boostedTree.trctrl2,
                                    tuneGrid=tuneGrid.gbm4,
                                    verbose=T)
)
#Variable importance from GBM
plot(boostedTree_3outcomes3)
plot(varImp(boostedTree_3outcomes3))
gbm.varimp <- varImp(boostedTree_3outcomes3, scale=T)
gbm.varimp_df <- as.data.frame(gbm.varimp[['importance']])
gbm.varimp_df <- mutate(gbm.varimp_df, variable=rownames(gbm.varimp_df))
rownames(gbm.varimp_df) <- 1:140

#Combination of variables from RF and GBM
morethan5imp.rf <- as.data.frame(morethan5imp.rf)
colnames(morethan5imp.rf) <- 'variable'
morethanzeroimp.gbm <- filter(gbm.varimp_df, Overall>0) %>% select(variable)
gbm.fr.var.selection <- unique(rbind(morethan5imp.rf, morethanzeroimp.gbm))
gbm.fr.var.selection <- as.vector(gbm.fr.var.selection$variable)

#Apply to train dataset
gbm.fr.var.selection_nozero_manual_grams_train <- nozero_manual_grams_train[,colnames(nozero_manual_grams_train)%in%gbm.fr.var.selection]
gbm.fr.var.selection_nozero_manual_grams_train <- cbind(TBQ=nozero_manual_grams_train$TBQ, ID_PACIENTE=nozero_manual_grams_train$ID_PACIENTE, gbm.fr.var.selection_nozero_manual_grams_train)

##New variable selection
tuneGrid.gbm5 <- expand.grid(n.trees=c(500),
                             interaction.depth=c(7:9),
                             shrinkage=c(0.1),
                             n.minobsinnode=10)

boostedTree.trctrl3 <- trainControl(method='cv', 
                                    number=3, 
                                    classProbs = T, 
                                    summaryFunction = multiClassSummary, 
                                    verboseIter=T,
                                    allowParallel=T)

system.time(
    boostedTree_3outcomes4 <- train(data=gbm.fr.var.selection_nozero_manual_grams_train,
                                    TBQ~. -ID_PACIENTE,
                                    distribution='multinomial',
                                    method='gbm',
                                    metric='Accuracy',
                                    trControl=boostedTree.trctrl3,
                                    tuneGrid=tuneGrid.gbm5,
                                    verbose=T)
)

gbm.test_results <- predict(boostedTree_3outcomes4, nozero_manual_grams_test)
confusionMatrix(gbm.test_results , nozero_manual_grams_test$TBQ)

########################################################
#Fitting model on new variable selection from GBM and RF
########################################################

#Creating selected dataset
var.selection_nozero_manual_grams <- nozero_manual_grams[,colnames(nozero_manual_grams)%in%gbm.fr.var.selection]
var.selection_nozero_manual_grams <- cbind(TBQ=nozero_manual_grams$TBQ, ID_PACIENTE=nozero_manual_grams$ID_PACIENTE, var.selection_nozero_manual_grams)

#Random forests
rf.trctrl.cv <- trainControl(method='cv', number=10, classProbs = T, summaryFunction = multiClassSummary,
                             allowParallel = F, verboseIter=T)

mtry_grid <- expand.grid(mtry=c(sqrt(82), sqrt(82)*3, sqrt(82)*4, sqrt(82)*5))

rf.nozero_manual_grams <- train(data=var.selection_nozero_manual_grams,
                                TBQ~. -ID_PACIENTE,
                                method='parRF',
                                trControl=rf.trctrl.cv,
                                tuneGrid=mtry_grid,
                                metric='Accuracy',
                                maximize=T,
                                importance=T,
                                ntree=1000,
                                verbose=T
)

plot(rf.nozero_manual_grams)

rf.test_results <- predict(rf.nozero_manual_grams, nozero_manual_grams_test)
confusionMatrix(rf.test_results, nozero_manual_grams_test$TBQ)
#0.96

#mtry=sqrt(82)*4


###############
##Boosted trees
###############

tuneGrid.gbm5 <- expand.grid(n.trees=c(300,400,500,600),
                             interaction.depth=c(8),
                             shrinkage=c(0.1),
                             n.minobsinnode=c(10,20,30))

boostedTree.trctrl3 <- trainControl(method='cv', 
                                    number=10, 
                                    classProbs = T, 
                                    summaryFunction = multiClassSummary, 
                                    verboseIter=T,
                                    allowParallel=T)

system.time(
        gbm_1 <- train(data=var.selection_nozero_manual_grams,
                                    TBQ~. -ID_PACIENTE,
                                    distribution='multinomial',
                                    method='gbm',
                                    metric='Accuracy',
                                    trControl=boostedTree.trctrl3,
                                    tuneGrid=tuneGrid.gbm5,
                                    verbose=T)
)

plot(gbm_1)
gbm_1.test_results <- predict(gbm_1, nozero_manual_grams_test)
confusionMatrix(gbm_1.test_results, nozero_manual_grams_test$TBQ)

#Acc 0.9584

###########
#SVM
###########

ctrl <- trainControl(method="cv",   # 10fold cross validation
                     repeats=10,		    # do 5 repititions of cv
                     summaryFunction=multiClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     verboseIter=T, 
                     allowParallel=F)
system.time(
svm.tune <- train(data=var.selection_nozero_manual_grams,
                  TBQ~. -ID_PACIENTE,
                  method = "svmRadial",
                  metric="Accuracy",
                  tuneLength = 9,
                  trControl=ctrl,
                  verbose=T)
)

grid <- expand.grid(sigma =seq(0.0001,0.01, by=0.002),
                    C =c(1,2,3))

system.time(
    svm.tune <- train(data=var.selection_nozero_manual_grams,
                      TBQ~. -ID_PACIENTE,
                      method = "svmRadial",
                      metric="Accuracy",
                      tuneGrid = grid,
                      trControl=ctrl,
                      verbose=T)
)

plot(svm.tune)

grid2 <- expand.grid(sigma =seq(0.008,0.03, by=0.002),
                    C =c(3))

system.time(
    svm.tune2 <- train(data=var.selection_nozero_manual_grams,
                      TBQ~. -ID_PACIENTE,
                      method = "svmRadial",
                      metric="Accuracy",
                      tuneGrid = grid2,
                      trControl=ctrl,
                      verbose=T)
)

plot(svm.tune2)

grid3 <- expand.grid(sigma =0.01,
                     C =seq(0.5, 4, by=0.5))

system.time(
    svm.tune3 <- train(data=var.selection_nozero_manual_grams,
                       TBQ~. -ID_PACIENTE,
                       method = "svmRadial",
                       metric="Accuracy",
                       tuneGrid = grid3,
                       trControl=ctrl,
                       verbose=T)
)

plot(svm.tune3)
svm.tune3.test_results <- predict(svm.tune3, nozero_manual_grams_test)
confusionMatrix(svm.tune3.test_results, nozero_manual_grams_test$TBQ)

#################
#Boosted logistic
#################

grid.log <- expand.grid(nIter=c(100,200,300,400,500))

ctrl2 <- trainControl(method="cv",   
                     repeats=10,		    
                     summaryFunction=multiClassSummary,	
                     classProbs=TRUE,
                     verboseIter=T, 
                     allowParallel=T)
system.time(
boost.log <- train(TBQ~. -ID_PACIENTE,
                        data=var.selection_nozero_manual_grams,
                        method = "LogitBoost",
                        metric="Accuracy",
                        tuneGrid=grid.log,
                        trControl=ctrl2,
                        verbose=T)
)
plot(boost.log)
boost.log_results <- predict(boost.log, nozero_manual_grams_test)
confusionMatrix(boost.log_results, nozero_manual_grams_test$TBQ)

grid.log2 <- expand.grid(nIter=c(500, 600, 700))

system.time(
    boost.log2 <- train(TBQ~. -ID_PACIENTE,
                       data=var.selection_nozero_manual_grams,
                       method = "LogitBoost",
                       metric="Accuracy",
                       tuneGrid=grid.log2,
                       trControl=ctrl2,
                       verbose=T)
)
plot(boost.log2)
boost.log_results2 <- predict(boost.log2, nozero_manual_grams_test)
confusionMatrix(boost.log_results2, nozero_manual_grams_test$TBQ)

#############
#Neural Network
#############

grid5 <- expand.grid(size=)

ctrl3 <- trainControl(method="cv",   
                      repeats=10,		    
                      summaryFunction=multiClassSummary,	
                      classProbs=TRUE,
                      verboseIter=T, 
                      allowParallel=F)

nnet.tune <- train(TBQ~. -ID_PACIENTE,
                   data=var.selection_nozero_manual_grams,
                   method = "nnet",
                   metric="Accuracy",
                   trControl=ctrl3,
                   tuneLength=5,
                   verbose=T)

plot(nnet.tune)
nnet.tune_results <- predict(nnet.tune, nozero_manual_grams_test)
confusionMatrix(nnet.tune_results, nozero_manual_grams_test$TBQ)

###############
# Bagged MARS
###############
grid6 <- expand.grid(size=)

ctrl4 <- trainControl(method="cv",   
                      repeats=10,		    
                      summaryFunction=multiClassSummary,	
                      classProbs=TRUE,
                      verboseIter=T, 
                      allowParallel=T)

bagged.MARS <- train(TBQ~. -ID_PACIENTE,
                   data=var.selection_nozero_manual_grams,
                   method = "bagEarth",
                   glm=list(family='multinomial'),
                   metric="Accuracy",
                   tuneLength=1,
                   trControl=ctrl4)

plot(bagged.MARS)
boost.gam_results <- predict(bagged.MARS, nozero_manual_grams_test)
confusionMatrix(bagged.MARS_results, nozero_manual_grams_test$TBQ)

###################
#Validation
###################

#Ensamble of algorithms
#Escribir el codigo del ensamble para el training set, setear el numero de voto q logran la mejor precision
ensamble_training <- select(manual_grams, ID_PACIENTE, FECHA, TBQ)
ensamble_training$RF <- predict(rf.nozero_manual_grams, nozero_manual_grams)
ensamble_training$GBM <- predict(gbm_1, nozero_manual_grams)
ensamble_training$SVM <- predict(svm.tune3, nozero_manual_grams)
ensamble_training$BLR <- predict(boost.log2, nozero_manual_grams)
ensamble_training$NN <- predict(nnet.tune, nozero_manual_grams)


#Probar con las 6000 marginales de los ptes de 2015 (TBQ - Training - evol marginales - pma0719336_evol_rnd6000_tbq)



#2005
#Evoluciones + en el alg screening


#2015-2016
#Evoluciones marginales
#Evoluciones + en el alg screening (Pasarlas por el alg de screening y tmb probar directo el ensamble)