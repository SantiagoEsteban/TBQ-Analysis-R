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


###################
#Validation
###################

#Ensamble of algorithms
#Escribir el codigo del ensamble para el training set, setear el numero de voto q logran la mejor precision
ensamble_training <- select(manual_grams, ID_PACIENTE, FECHA, TBQ)
ensamble_training$RF <- predict(rf.nozero_manual_grams, nozero_manual_grams) #0.957
#ensamble_training$GBM <- predict(gbm_1, nozero_manual_grams) #0.9496
ensamble_training$SVM <- predict(svm.tune3, nozero_manual_grams) #0.959
#ensamble_training$PLR <- predict(penalized.log, nozero_manual_grams) #0.955
ensamble_training$NN <- predict(nnet.tune, nozero_manual_grams) #0.958
ensamble_training$TBQ <- make.names(ensamble_training$TBQ)
#t_ensamble_training <- select(ensamble_training, -ID_PACIENTE, -FECHA, -TBQ) %>% t()

#########
#SVM
#########
ctrl <- trainControl(method="cv",   
                     repeats=5,		    
                     summaryFunction=multiClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     verboseIter=T, 
                     allowParallel=F)

grid.blended.linear <- expand.grid(C=c(0.001, 0.01, 0.02, 0.03))

system.time(
    svm.blended.linear <- train(data=select(ensamble_training, -ID_PACIENTE, -FECHA),
                         TBQ~.,
                         method = "svmLinear",
                         metric="Accuracy",
                         tuneGrid = grid.blended.linear,
                         trControl=ctrl,
                         verbose=T)
)
#Acc0.966, C=0.03
ensamble_training$Final <- predict(svm.blended.linear, ensamble_training)



####TRY PCA
#Use screeplot to decide #of prcomp
#Use algorithms with PCA
PCA <- preProcess(select(nozero_manual_grams, -ID_PACIENTE, -TBQ), method = c("pca"))
plot(PCA)
nozero_manual_grams_pca <- predict(PCA, nozero_manual_grams)
plot(princomp(select(nozero_manual_grams, -ID_PACIENTE, -TBQ)), 
          type = 'lines', npcs=70, 
     main="Scree Plot for Principal Components vs Variance Explained")
abline(h=0, col="red")


#Probar con las 6000 marginales de los ptes de 2015 (TBQ - Training - evol marginales - pma0719336_evol_rnd6000_tbq)
marginal2015 <- as.data.frame(read_excel('TBQ - Training - evol marginales - pma0719336_evol_rnd6000_tbq_var.xlsx'))
marginal2015 <- filter(marginal2015, ID_PACIENTE!=" ")

ensamble_marginal2015 <- select(marginal2015, ID_PACIENTE, FECHA_CARGA, TBQ)
ensamble_marginal2015$RF <- predict(rf.nozero_manual_grams, marginal2015)
ensamble_marginal2015$SVM <- predict(svm.tune3, marginal2015) 
#ensamble_marginal2015$PLR <- predict(penalized.log, marginal2015)
ensamble_marginal2015$NN <- predict(nnet.tune, marginal2015)
ensamble_marginal2015$Final <- predict(svm.blended.linear, ensamble_marginal2015)
confusionMatrix(ensamble_marginal2015$Final, make.names(ensamble_marginal2015$TBQ))
# ACC 0.9917



#2005
#Evoluciones + en el alg screening

postscreening2005 <- as.data.frame(read_excel('TBQBOFW_2005_Total_Validado_var.xlsx'))
postscreening2005 <- filter(postscreening2005, ID_PACIENTE!=" ")


ensamble_postscreening2005 <- select(postscreening2005, ID_PACIENTE, FECHA, TBQ)
ensamble_postscreening2005$RF <- predict(rf.nozero_manual_grams, postscreening2005) 
ensamble_postscreening2005$SVM <- predict(svm.tune3, postscreening2005) 
ensamble_postscreening2005$NN <- predict(nnet.tune, postscreening2005)
ensamble_postscreening2005$Final <- predict(svm.blended.linear, ensamble_postscreening2005)
confusionMatrix(ensamble_postscreening2005$Final, make.names(ensamble_postscreening2005$TBQ))

#ACC 0.9532

#Wide results by occasion(t)
ensamble_postscreening2005_t <- filter(ensamble_postscreening2005, FECHA >= 1/1/2000) %>% 
    unique() %>% group_by(ID_PACIENTE) %>% mutate(t=rank(FECHA, ties.method='random'))
ensamble_postscreening2005_wide <- select(ensamble_postscreening2005_t, ID_PACIENTE, final,t) %>% 
    spread(t, final)

#Wide results by year
library(lubridate)
#By year table
ensamble_postscreening2005_year <- ensamble_postscreening2005 %>% filter(FECHA >= '2000/1/1' & final!='X9') %>%
    group_by(ID_PACIENTE) %>% mutate(t=rank(FECHA, ties.method='random'))
ensamble_postscreening2005_year$FECHA <- floor_date(ensamble_postscreening2005_year$FECHA, unit='year')
ensamble_postscreening2005_year <- group_by(ensamble_postscreening2005_year, ID_PACIENTE, FECHA) %>% filter(t==max(t))
ensamble_postscreening2005_year <- select(ensamble_postscreening2005_year, ID_PACIENTE, Final,FECHA) %>% 
    spread(FECHA, Final)

#Graphing trajectories
IDs <- c(66714, 32969, 62638, 63491, 66714, 203943, 78737)

ggplot(filter(ensamble_postscreening2005, ID_PACIENTE%in%IDs & Final!='X9'), aes(x=FECHA, y=Final, group=as.factor(ID_PACIENTE), color=as.factor(ID_PACIENTE))) + 
    geom_point() + 
    geom_path(size=1) + 
    guides(color=FALSE) + 
    theme_gray() + 
    facet_wrap(~ID_PACIENTE, scale='free') +
    ggtitle('Smoking status vs Time') +
    xlab('Time') + ylab('Smoking Status')
    
#2015-2016
#Evoluciones marginales
marginal2016 <- as.data.frame(read_excel('pma0719336_evol_rnd6000_tst_tbq_var1.xlsx'))
marginal2016 <- filter(marginal2016, ID_PACIENTE!=" ")

ensamble_marginal2016 <- select(marginal2016, ID_PACIENTE, FECHA_CARGA, TBQ)
ensamble_marginal2016$RF <- predict(rf.nozero_manual_grams, marginal2016)
ensamble_marginal2016$SVM <- predict(svm.tune3, marginal2016) 
ensamble_marginal2016$NN <- predict(nnet.tune, marginal2016)
ensamble_marginal2016$Final <- predict(svm.blended.linear, ensamble_marginal2016)
confusionMatrix(ensamble_marginal2016$Final, make.names(ensamble_marginal2016$TBQ))


#Acc 0.9908

#Evoluciones + en el alg screening (Pasarlas por el alg de screening y tmb probar directo el ensamble)
postscreening2016 <- as.data.frame(read_excel('Filtrado_2015_2016_pma0719336_evol_full_tst_tbq_problemas_var.xlsx'))
postscreening2016 <- filter(postscreening2016, ID_PACIENTE!=" ")


ensamble_postscreening2016 <- select(postscreening2016, ID_PACIENTE, FECHA, TBQ)
ensamble_postscreening2016$RF <- predict(rf.nozero_manual_grams, postscreening2016) 
ensamble_postscreening2016$SVM <- predict(svm.tune3, postscreening2016) 
ensamble_postscreening2016$NN <- predict(nnet.tune, postscreening2016)
ensamble_postscreening2016$Final <- predict(svm.blended.linear, ensamble_postscreening2016)
confusionMatrix(ensamble_postscreening2016$Final, make.names(ensamble_postscreening2016$TBQ))
#ACC 0.9266


#Algoritmo para los missing
#BASE DE PTES MISSING.
##############
missing <- read_excel('notbqdata2015completo.xlsx')
missing <- filter(missing, ID_PACIENTE!=" ")
algunavez <- read_excel('llamadosnodatatbq.xlsx')
algunavez <- filter(algunavez, ID_PACIENTE!=" ") %>% select(ID_PACIENTE, fumoalgunavez)
missing2 <- tbl_df(read_excel('PMA0719336_cv.xlsx')) %>% select(ID_PACIENTE, antec_cv) %>% full_join(missing, by='ID_PACIENTE')
missing2 <- missing2[!duplicated(missing2),]
missing3 <- tbl_df(read_excel('PMA0719336_cerebv.xlsx')) %>% select(ID_PACIENTE, antec_cerebro) %>% full_join(missing2, by='ID_PACIENTE')
missing3 <- missing3[!duplicated(missing3),]
only_eversmokers <- ensamble_training %>% filter(TBQ!='X9') %>% filter(TBQ!='X0') %>% select(ID_PACIENTE, TBQ) %>% unique()
colnames(only_eversmokers) <- c('ID_PACIENTE', 'fumoalgunavez')
only_eversmokers$fumoalgunavez <- 1
algunavez2 <- full_join(algunavez, only_eversmokers, by='ID_PACIENTE')
algunavez3 <- algunavez2 %>% unite(fumoalgunavez, c(fumoalgunavez.x, fumoalgunavez.y))
algunavez3$fumoalgunavez <- gsub('_NA', x=algunavez3$fumoalgunavez, ' ')
algunavez3$fumoalgunavez <- gsub('NA_', x=algunavez3$fumoalgunavez, ' ')
algunavez3$fumoalgunavez <- gsub('0_', x=algunavez3$fumoalgunavez, ' ')
algunavez3$fumoalgunavez <- gsub('1_', x=algunavez3$fumoalgunavez, ' ')
algunavez3$fumoalgunavez <- as.numeric(algunavez3$fumoalgunavez)
algunavez3$fumoalgunavez <- make.names(algunavez3$fumoalgunavez)
missing3 <- full_join(missing3, algunavez3, by='ID_PACIENTE')
missing3$fumoalgunavez[is.na(missing3$fumoalgunavez)] <- 'X0'
missing3 <- make.names(missing3$tbqact2015)

#Analyzing frequencies
freq_missing3 <- colSums(missing3, na.rm = T)
sum(freq_missing3 > 0) #No 0 columns

#ZeroVar
#nzvarmissing <- nearZeroVar(missing3)
#missing4 <- select(missing3, -nzvarmissing)

#Analyzing correlation
missing3_cor <- cor(select(missing3, -ID_PACIENTE, -tbqact2015, -antecedentetotal))
library(corrplot)
corrplot(missing3_cor, order = "hclust", tl.cex=0.6)
#Finding highly correlated variables
highCorr <- findCorrelation(missing3_cor, cutoff = .75) #Solo MED_EVOL_PREQX y PREQX
missing4 <- select(missing3, -highCorr)
missing4$tbqact2015 <- make.names(missing4$tbqact2015)

missing5 <- filter(missing4, tbqact2015!='NA.') %>% select(-antecedentetotal)

missing_NoRegTBQ_PCA <- filter(missing5, RegTBQ==0) %>% select(-RegTBQ, -MED_EVOL_UTIA)

#PCA
#Use screeplot to decide #of prcomp
#Use algorithms with PCA
PCA_full <- preProcess(select(missing5, -ID_PACIENTE, -tbqact2015), method = c("pca"))
PCA_NoRegTBQ <- preProcess(select(missing_NoRegTBQ_PCA, -ID_PACIENTE, -tbqact2015), method = c("pca"))

missing5_pca <- predict(PCA, missing5)
PCA_missing <-princomp(select(missing5, -ID_PACIENTE, -tbqact2015))
plot(PCA_missing, type = 'lines', npcs=50, main="Scree Plot for Principal Components vs Variance Explained")
abline(h=0, col="red")
summary(PCA_missing)

missing_NoRegTBQ_PCA <- predict(PCA_NoRegTBQ, missing_NoRegTBQ_PCA)
PCA_missing_NoRegTBQ_PCA <-princomp(select(missing_NoRegTBQ_PCA, -ID_PACIENTE, -tbqact2015))
plot(PCA_missing_NoRegTBQ_PCA, type = 'lines', npcs=50, main="Scree Plot for Principal Components vs Variance Explained")
abline(h=0, col="red")
summary(PCA_missing)


ctrl <- trainControl(method="cv",   
                     number=10,		    
                     summaryFunction=twoClassSummary,
                     classProbs=TRUE,
                     verboseIter=T, 
                     allowParallel=F)

grid.svm <- expand.grid(sigma=c(0.01,0.05,1),C=2)

system.time(
    svm.missing_pca <- train(data=select(missing5, -ID_PACIENTE),
                             tbqact2015~.,
                     method = "svmRadial",
                     metric="ROC",
                     tuneLength=5,
                     trControl=ctrl,
                     verbose=T)
)

svm_missing_pca_results <- predict(svm.missing_pca, missing_NoRegTBQ_PCA)
confusionMatrix(svm_missing_pca_results, missing_NoRegTBQ_PCA$tbqact2015)



###
#SVM
####

ctrl <- trainControl(method="cv",   
                     repeats=10,		    
                     summaryFunction=multiClassSummary,	
                     classProbs=TRUE,
                     verboseIter=T, 
                     allowParallel=F)
system.time(
    svm.tune <- train(data=filter(missing3, tbqact2015!=NA) %>% select(-ID_PACIENTE, -tbqact2015),
                      make.names(fumoalgunavez)~.,
                      method = "svmRadial",
                      metric="Accuracy",
                      tuneLength = 9,
                      trControl=ctrl,
                      verbose=T)
)

grid <- expand.grid(sigma =seq(0.0001,0.01, by=0.002),
                    C =c(1,2,3))
