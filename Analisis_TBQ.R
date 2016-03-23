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
cl<-makeCluster(2) #change the 4 to your number of CPU cores
registerDoSNOW(cl)
#library(doParallel)

##################
#Import manual dataset created with REGEX SAS
##################

manual_grams <- as.data.frame(read_excel('tbqcompleto2015_FINAL_nodups2.xlsx'))
manual_grams <- select(manual_grams, -FECHA, -TEXTO)

#Analyzing frequencies
freq_manual_grams <- colSums(manual_grams)
nozero_freq_manual_grams <- freq_manual_grams > 0
sum(nozero_freq_manual_grams)
nozero_manual_grams <- manual_grams[, nozero_freq_manual_grams]

#Near zero var
nearZeroVar(nozero_manual_grams)

#Correlations
manual_grams_cor <- cor(select(nozero_manual_grams, -TBQ, -notbqdata, -ID_PACIENTE, -notbqvar, -total))
#corrplot(manual_grams_cor, order = "hclust", tl.cex=0.5)
#Finding highly correlated variables
findCorrelation(manual_grams_cor, cutoff = .75) #no correlations above 0.75

#############
#CORPUS
#############


corp <- read_excel('tbqcompleto2015_validar3_nodups1.xlsx')
corpatrib <- data.frame(cbind(corp[,1:2], corp[,4], seq(1:length(corp[,1]))))
colnames(corpatrib) <- c("ID_PACIENTE", "FECHA", "TBQ", "ID_UNIQUE")
corptext <- as.character(corp$TEXTO)
names(corptext) <- corpatrib$ID_UNIQUE
corpus1 <- Corpus(VectorSource(corptext))
#shortcorpus <- Corpus(VectorSource(corptext[1:10]))

#Apply custom transformations with tm package
#create the toSpace content transformer
toSpace <- content_transformer(function(x, pattern){
    return(gsub(pattern," ",x))
})
corpus2 <- tm_map(corpus1, toSpace, ":")
corpus2 <- tm_map(corpus2, toSpace, "\\.")
corpus2 <- tm_map(corpus2, toSpace, "_x000D_\r\n")
corpus2 <- tm_map(corpus2, toSpace, ",")
corpus2 <- tm_map(corpus2, toSpace, ";")
corpus2 <- tm_map(corpus2, toSpace, "\\?")
corpus2 <- tm_map(corpus2, toSpace, "!")
corpus2 <- tm_map(corpus2, toSpace, ",")
#corpus2 <- tm_map(corpus2, toSpace, " l ")
#corpus2 <- tm_map(corpus2, removePunctuation(corpus2, preserve_intra_word_dashes = TRUE))
corpus2 <- tm_map(corpus2, content_transformer(tolower))
corpus2 <- tm_map(corpus2, removeNumbers)
corpus2 <- tm_map(corpus2, stripWhitespace)

#shortcorpus2 <- tm_map(shortcorpus, toSpace, ":")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, "\\.")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, "_x000D_\r\n")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, ",")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, ";")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, "\\?")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, "!")
#shortcorpus2 <- tm_map(shortcorpus2, toSpace, " l ")
#shortcorpus2 <- tm_map(shortcorpus2, removePunctuation(preserve_intra_word_dashes = TRUE))
#shortcorpus2 <- tm_map(shortcorpus2, content_transformer(tolower))
#shortcorpus2 <- tm_map(shortcorpus2, removeNumbers)
#shortcorpus2 <- tm_map(shortcorpus2, stripWhitespace)

#Steps: 1) tokenize 
#       2) remove all non related words(will get this from sas variables created manually)
#       3) ngrams 1:4
#       4) Stem?
#       5) DocumentTermMatrix (see below)
#       5) Analyze according to A gentle introduction to text mining using R and RTextTools: A Supervised Learning Package for Text Classification


#Step 1
#Create tbq terms list based on the qualitative analysis of clinical notes
tbqterms <- unique(as.character(read.csv('Terminos TBQ.txt', header=FALSE)$V1)) #TBQ TERMS

###################
#FOR N-GRAM VERSION
###################
#Function that eliminates all non-listed words
keepOnlyWords<-content_transformer(function(x,words) {
    regmatches(x, 
               gregexpr(paste0("\\b(",  paste(words,collapse="|"),"\\b)"), x)
               , invert=T)<-" "
    x
})

#Removing non-listed words
corpus2 <- tm_map(corpus2, keepOnlyWords, tbqterms)
corpus2 <- tm_map(corpus2, stripWhitespace)
corpus2 <- tm_map(corpus2, content_transformer(tolower))

#Creating 4-grams
FourgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 4))
Four_gram_tdm <- DocumentTermMatrix(corpus2, control=list(tokenize=FourgramTokenizer, 
                                                weighting=function(x) weightTfIdf(x, normalize =FALSE)))

#Tidy up Four_gram_tdm
#Removing sparse terms
Four_gram_tdm
Four_gram_tdm_nosparse <- removeSparseTerms(Four_gram_tdm, 0.989)

#removing zero variance terms
Four_gram_tdm_df <- tbl_df(as.data.frame(as.matrix(Four_gram_tdm_nosparse)))
nearZeroVar(Four_gram_tdm_df) #too many

nozero_Four_gram_tdm_df <- colSums(Four_gram_tdm_df) > 0
Four_gram_tdm_df <- Four_gram_tdm_df[, nozero_Four_gram_tdm_df]

#removing highly correlated
par(mfrow=c(1,1))
Four_gram_tdm_df_cor <- cor(Four_gram_tdm_df)
#corrplot(Four_gram_tdm_df_cor, order = "hclust", tl.cex=0.2)
#Finding highly correlated variables
highCorr <- findCorrelation(Four_gram_tdm_df_cor, cutoff = .75)
highCorr
Four_gram_tdm_df_nocor <- Four_gram_tdm_df[, -highCorr]
#corrplot(cor(Four_gram_tdm_df_nocor), order = "hclust", tl.cex=0.2)

#Adding outcome
Four_gram_tdm_outcome <- cbind(manual_grams$TBQ, Four_gram_tdm_df_nocor) 

#LISTO HASTA ACA N-GRAMS


##############################
#Generar tdm solo con los terminos sueltos, sin n-grams
##############################
one_gram_tdm_tf <- DocumentTermMatrix(corpus2, control=list(wordLengths=c(1, Inf), dictionary=tbqterms))
one_gram_tdm_tfidf <- DocumentTermMatrix(corpus2, control=list(wordLengths=c(1, Inf), dictionary=tbqterms, weighting=function(x) weightTfIdf(x, normalize =FALSE)))

#Analyzing frequencies
one_gram_tdm_tf_df <- tbl_df(as.data.frame(as.matrix(one_gram_tdm_tf)))
nozero_freq_one_gram_tdm_tf_df <- colSums(one_gram_tdm_tf_df) > 0
one_gram_tdm_tf_df <- one_gram_tdm_tf_df[,nozero_freq_one_gram_tdm_tf_df]

one_gram_tdm_tfidf_df <- tbl_df(as.data.frame(as.matrix(one_gram_tdm_tfidf)))
nozero_freq_one_gram_tdm_tfidf_df <- colSums(one_gram_tdm_tfidf_df) > 0
one_gram_tdm_tfidf_df <- one_gram_tdm_tfidf_df[,nozero_freq_one_gram_tdm_tfidf_df]

#removing highly correlated
one_gram_tdm_tf_df_cor <- cor(one_gram_tdm_tf_df)
#corrplot(one_gram_tdm_tf_df_cor, order = "hclust", tl.cex=0.2)
#Finding highly correlated variables
highCorr <- findCorrelation(one_gram_tdm_tf_df_cor, cutoff = .75)
highCorr

one_gram_tdm_tfidf_df_cor <- cor(one_gram_tdm_tfidf_df)
#corrplot(one_gram_tdm_tfidf_df_cor, order = "hclust", tl.cex=0.2)
#Finding highly correlated variables
highCorr <- findCorrelation(one_gram_tdm_tfidf_df_cor, cutoff = .75)
highCorr

#Adding outcome
one_gram_tdm_tfidf_outcome <- cbind(manual_grams$TBQ, one_gram_tdm_tfidf_df)
one_gram_tdm_tfidf_outcome <- cbind(manual_grams$notbqdata, one_gram_tdm_tfidf_outcome)
one_gram_tdm_tf_df_outcome <- cbind(manual_grams$TBQ, one_gram_tdm_tf_df)
one_gram_tdm_tf_df_outcome <- cbind(manual_grams$notbqdata, one_gram_tdm_tf_df_outcome)

###HASTA ACA One-GRAMS con TF weights and TF/ITF

##################
#Import manual dataset created with REGEX SAS
##################

manual_grams <- as.data.frame(read_excel('tbqcompleto2015_FINAL_nodups2.xlsx'))
manual_grams <- select(manual_grams, -FECHA, -TEXTO)

#Analyzing frquencies
freq_manual_grams <- colSums(manual_grams)
nozero_freq_manual_grams <- freq_manual_grams > 0
sum(nozero_freq_manual_grams)
nozero_manual_grams <- manual_grams[, nozero_freq_manual_grams]

#Near zero var
nearZeroVar(nozero_manual_grams)

#Correlations
manual_grams_cor <- cor(select(nozero_manual_grams, -TBQ, -notbqdata, -ID_PACIENTE, -notbqvar, -total))
#corrplot(manual_grams_cor, order = "hclust", tl.cex=0.5)
#Finding highly correlated variables
findCorrelation(manual_grams_cor, cutoff = .75) #no correlations above 0.75

###############
#BASE DE PTES MISSING.
##############
missing <- read_excel('notbqdata2015completo.xlsx')
missing <- tbl_df(missing[1:1819,])

missing2 <- tbl_df(read_excel('PMA0719336_cv.xlsx')) %>% select(ID_PACIENTE, antec_cv) %>% full_join(missing, by='ID_PACIENTE')
missing2 <- missing2[!duplicated(missing2),]
missing3 <- tbl_df(read_excel('PMA0719336_cerebv.xlsx')) %>% select(ID_PACIENTE, antec_cerebro) %>% full_join(missing2, by='ID_PACIENTE')
missing3 <- missing3[!duplicated(missing3),]

#Analyzing frequencies
freq_missing3 <- colSums(missing3, na.rm = T)
nozero_freq_missing3 <- freq_missing3 > 0 #No 0 columns

#Analyzing correlation
missing3_cor <- cor(select(missing3, -ID_PACIENTE, -tbqact2015, -antecedentetotal, -totalevolprimera, -totalevoluciones, -sumantecedentetotal, -total_int))
#corrplot(missing3_cor, order = "hclust", tl.cex=0.6)
#Finding highly correlated variables
highCorr <- findCorrelation(missing3_cor, cutoff = .75) #Solo MED_EVOL_PREQX y PREQX

###############
#ANALYSIS PLAN
###############
# Algorimot por separado para "evol sin datos sobre tbq"? o un solo algoritmo con 3 outcome con 3 niveles?
# Evaluar los algoritmos para clasificacion: rand forests, boosted trees, SVM, neural networks, C5.

#Random Forests
#nozero_manual_grams
#one_gram_tdm_tfidf_outcome 
#one_gram_tdm_tf_outcome
#Four_gram_tdm_outcome
#missing3

#Manual grams
library(caret)
set.seed(123)
length(nozero_manual_grams)
nozero_manual_grams$TBQ <- as.factor(nozero_manual_grams$TBQ)
rf.trctrl.oob <- trainControl(method='oob', classProbs = T, summaryFunction = multiClassSummary,
                              allowParallel = T, verboseIter=T)
rf.trctrl.boot632 <- trainControl(method='boot632', classProbs = T, summaryFunction = multiClassSummary,
                              allowParallel = T, verboseIter=T)

rf.trctrl.cv <- trainControl(method='cv', number=10, classProbs = T, summaryFunction = multiClassSummary,
                                  allowParallel = T, verboseIter=T)

mtry_grid <- expand.grid(mtry=c(1, sqrt(142), log(143), log(142/2)))
mtry_grid2 <- expand.grid(mtry=c(sqrt(142))
mtry_grid3 <- expand.grid(mtry=c(sqrt(142)*2, sqrt(142)*3, sqrt(142)*4))

nozero_manual_grams_TBQ <- filter(nozero_manual_grams, notbqdata==0) %>% select(-notbqdata, -ID_PACIENTE)
nozero_manual_grams_TBQ$TBQ <- as.factor(factor(nozero_manual_grams_TBQ$TBQ))
nozero_manual_grams_TBQ$TBQ <- make.names(nozero_manual_grams_TBQ$TBQ)


TBQ_train <- createDataPartition(nozero_manual_grams_TBQ$TBQ,
                                                     p=.8,
                                                     list=F,
                                                     times=1)
nozero_manual_grams_TBQ_train <- nozero_manual_grams_TBQ[TBQ_train,]
nozero_manual_grams_TBQ_test <- nozero_manual_grams_TBQ[-TBQ_train,]

rf.oob.nozero_manual_grams <- train(data=nozero_manual_grams_TBQ_train,
                                    TBQ~.,
                                    method='parRF',
                                    trControl=rf.trctrl.oob,
                                    tuneGrid=mtry_grid,
                                    metric='ROC',
                                    maximize=T,
                                    importance=T,
                                    ntree=1000
                                    )

rf.cv.nozero_manual_grams <- train(data=nozero_manual_grams_TBQ_train,
                                    TBQ~.,
                                    method='parRF',
                                    trControl=rf.trctrl.cv,
                                    tuneGrid=mtry_grid2,
                                    metric='ROC',
                                    maximize=T,
                                    importance=T,
                                    ntree=1000
)

rf.cv2.nozero_manual_grams <- train(data=nozero_manual_grams_TBQ_train,
                                   TBQ~.,
                                   method='parRF',
                                   trControl=rf.trctrl.cv,
                                   tuneGrid=mtry_grid3,
                                   metric='ROC',
                                   maximize=T,
                                   importance=T,
                                   ntree=1000
)
varImp.rf.cv2.nozero_manual_grams <- varImp(rf.cv2.nozero_manual_grams)
plot(varImp.rf.cv2.nozero_manual_grams)

test_results <- predict(rf.cv2.nozero_manual_grams, nozero_manual_grams_TBQ_test)
confusionMatrix(test_results, nozero_manual_grams_TBQ_test$TBQ)

#Manual_grams tbq data si o no

nozero_manual_grams_TBQornot<- select(nozero_manual_grams, -ID_PACIENTE, -TBQ)
nozero_manual_grams_TBQornot$notbqdata <- make.names(as.factor(nozero_manual_grams_TBQornot$notbqdata))

TBQornot_train <- createDataPartition(nozero_manual_grams_TBQornot$notbqdata,
                                 p=.8,
                                 list=F,
                                 times=1)
nozero_manual_grams_TBQornot_train <- nozero_manual_grams_TBQornot[TBQornot_train,]
nozero_manual_grams_TBQornot_test <- nozero_manual_grams_TBQornot[-TBQornot_train,]

rf.trctrl.cv.TBQornot <- trainControl(method='cv', number=2, classProbs = T, summaryFunction = multiClassSummary,
                             allowParallel = T, verboseIter=T)

mtry_grid_TBQornot <- expand.grid(mtry=c(sqrt(142), sqrt(142)*2, sqrt(142)*3, sqrt(142)*4))

rf.cv.nozero_manual_grams_TBQornot <- train(data=nozero_manual_grams_TBQornot_train,
                                    notbqdata~.,
                                    method='parRF',
                                    trControl=rf.trctrl.cv.TBQornot,
                                    tuneGrid=mtry_grid_TBQornot,
                                    metric='ROC',
                                    maximize=T,
                                    importance=T,
                                    ntree=1000
)

test_results2 <- predict(rf.cv.nozero_manual_grams_TBQornot, nozero_manual_grams_TBQornot_test)
confusionMatrix(test_results2, nozero_manual_grams_TBQornot_test$notbqdata)

#Manual_grams with three outcomes

nozero_manual_grams_3outcomes <- nozero_manual_grams
nozero_manual_grams_3outcomes$TBQ <- make.names(nozero_manual_grams_3outcomes$TBQ)

three_outcomes_train <- createDataPartition(nozero_manual_grams_3outcomes$TBQ,
                                      p=.8,
                                      list=F,
                                      times=1)
nozero_manual_grams_3outcomes_train <- nozero_manual_grams_3outcomes[three_outcomes_train,]
nozero_manual_grams_3outcomes_test <- nozero_manual_grams_3outcomes[-three_outcomes_train,]

rf.trctrl.cv.3outcomes <- trainControl(method='cv', number=2, classProbs = T, summaryFunction = multiClassSummary,
                                      allowParallel = T, verboseIter=T)

mtry_grid_3outcomes <- expand.grid(mtry=c(sqrt(142)))

rf.cv.nozero_manual_grams_3outcomes <- train(data=nozero_manual_grams_3outcomes_train,
                                            TBQ~. -ID_PACIENTE -notbqdata,
                                            method='parRF',
                                            trControl=rf.trctrl.cv.3outcomes,
                                            tuneGrid=mtry_grid_3outcomes,
                                            metric='Accuracy',
                                            maximize=T,
                                            importance=T,
                                            ntree=1000
)
test_results3 <- predict(rf.cv.nozero_manual_grams_3outcomes, nozero_manual_grams_3outcomes_test)
confusionMatrix(test_results3, nozero_manual_grams_3outcomes_test$TBQ)

#One_grams
#one_gram_tdm_tfidf_outcome 
#one_gram_tdm_tf_outcome

colnames(one_gram_tdm_tfidf_outcome)[1] <- 'notbqdata'
colnames(one_gram_tdm_tfidf_outcome)[2] <- 'TBQ'

one_gram_tdm_tfidf_outcome$TBQ <- make.names(as.factor(one_gram_tdm_tfidf_outcome$TBQ))
one_gram_tdm_tfidf_outcome$notbqdata <- make.names(as.factor(one_gram_tdm_tfidf_outcome$notbqdata))

one_grams_TBQornot_train <- createDataPartition(one_gram_tdm_tfidf_outcome$notbqdata,
                                      p=.8,
                                      list=F,
                                      times=1)
one_gram_tdm_tfidf_outcome_TBQornot_train <- one_gram_tdm_tfidf_outcome[one_grams_TBQornot_train,]
one_gram_tdm_tfidf_outcome_TBQornot_test <- one_gram_tdm_tfidf_outcome[-one_grams_TBQornot_train,]

#one_gram_tdm_tfidf_outcome_TBQornot_train <- one_gram_tdm_tfidf_outcome_TBQornot_train %>% select(-manual_grams$TBQ)


rf.trctrl.cv.TBQornot_one_gram <- trainControl(method='cv', number=2, classProbs = T, summaryFunction = multiClassSummary,
                                      allowParallel = T, verboseIter=T)

mtry_grid_TBQornot_one_gram <- expand.grid(mtry=c(log(119),sqrt(119), sqrt(119)*4))

rf.cv.one_gram_tdm_tfidf_outcome_TBQornot <- train(data=one_gram_tdm_tfidf_outcome_TBQornot_train,
                                            notbqdata~. -TBQ,
                                            method='parRF',
                                            trControl=rf.trctrl.cv.TBQornot_one_gram,
                                            tuneGrid=mtry_grid_TBQornot_one_gram,
                                            metric='ROC',
                                            maximize=T,
                                            importance=T,
                                            ntree=1000
)
importance(rf.cv.one_gram_tdm_tfidf_outcome_TBQornot)

#######################
# 4_grams
#######################


Four_gram_tdm_outcome_3outcomes <- Four_gram_tdm_outcome
colnames(Four_gram_tdm_outcome_3outcomes)[1] <- "TBQ" 
Four_gram_tdm_outcome_3outcomes$TBQ <- make.names(Four_gram_tdm_outcome_3outcomes$TBQ)

Four_gram_three_outcomes_train <- createDataPartition(Four_gram_tdm_outcome_3outcomes$TBQ,
                                            p=.8,
                                            list=F,
                                            times=1)
Four_gram_tdm_outcome_3outcomes_train <- Four_gram_tdm_outcome_3outcomes[Four_gram_three_outcomes_train,]
Four_gram_tdm_outcome_3outcomes_test <- Four_gram_tdm_outcome_3outcomes[-Four_gram_three_outcomes_train,]

rf.trctrl.cv.Four_gram <- trainControl(method='cv', number=2, classProbs = T, summaryFunction = multiClassSummary,
                                       allowParallel = T, verboseIter=T)

mtry_grid_Four_gram_3outcomes <- expand.grid(mtry=c(sqrt(439)))

rf.cv.our_gram_tdm_outcome_3outcomes <- train(data=Four_gram_tdm_outcome_3outcomes_train,
                                             TBQ~.,
                                             method='parRF',
                                             trControl=rf.trctrl.cv.Four_gram,
                                             tuneGrid=mtry_grid_Four_gram_3outcomes,
                                             metric='Accuracy',
                                             maximize=T,
                                             importance=T,
                                             ntree=1000
)
test_results3 <- predict(rf.cv.nozero_manual_grams_3outcomes, nozero_manual_grams_3outcomes_test)
confusionMatrix(test_results3, nozero_manual_grams_3outcomes_test$TBQ)

########################
#Boosting tree
########################

nozero_manual_grams_3outcomes_test
nozero_manual_grams_3outcomes_train

boostedTree.trctrl.cv.3outcomes <- trainControl(method='cv', 
                                                number=2, 
                                                classProbs = T, 
                                                summaryFunction = multiClassSummary, 
                                                verboseIter=T)

ntrees <- c(100,200,500,1000)
interaction_depth <- c(2:9)
shrink <- c(0.001, 0.01, 0.1)
tuneGrid.gbm <- expand.grid(n.trees=c(100, 500, 1000),
                            interaction.depth=c(2:9),
                            shrinkage=c(0.1),
                            n.minobsinnode=10)
system.time(
boostedTree_3outcomes <- train(data=nozero_manual_grams_3outcomes_train,
                               TBQ~. -notbqdata -ID_PACIENTE,
                               distribution='multinomial',
                               method='gbm',
                               trControl=boostedTree.trctrl.cv.3outcomes,
                               tuneGrid=tuneGrid.gbm,
                               verbose=T)
)



system.time(
boostedTree_3outcomes <- gbm(TBQ~. -notbqdata -ID_PACIENTE,
                             data=nozero_manual_grams_3outcomes_train,
                             distribution='multinomial',
                             n.trees=c(100, 500, 1000),
                             interaction.depth=c(2:9),
                             shrinkage=c(0.1),
                             n.minobsinnode=10,
                             cv.folds=10,
                             verbose=T,
                             n.cores=2)

)

boostedTree_3outcomes_pred <- predict(boostedTree_3outcomes, nozero_manual_grams_3outcomes_test)
confusionMatrix(boostedTree_3outcomes_pred, nozero_manual_grams_3outcomes_test$TBQ)
varImp(boostedTree_3outcomes)
boostedTree_3outcomes_predprob <- predict(boostedTree_3outcomes, nozero_manual_grams_3outcomes_test, type='prob')
boostedTree_3outcomes_predprob$obs <- nozero_manual_grams_3outcomes_test$TBQ
head(boostedTree_3outcomes_predprob)
plot(boostedTree_3outcomes)
plot(varImp(boostedTree_3outcomes))
####################
#SVM
####################

nozero_manual_grams_3outcomes_test
nozero_manual_grams_3outcomes_train

ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
                     repeats=5,		    # do 5 repititions of cv
                     summaryFunction=multiClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,verboseIter=T)

svm.tune <- train(TBQ~. -notbqdata -ID_PACIENTE -total,
                  data=nozero_manual_grams_3outcomes_train,
                  method = "svmRadial",
                  metric="Accuracy",
                  tuneLength = 9,
                  trControl=ctrl,
                  verbose=T)

grid <- expand.grid(sigma = c(seq(0.1,0.5, by=0.05)),
                    C = c(7.50, 8, 8.5, 9))

svm.tune2 <- train(TBQ~. -notbqdata -ID_PACIENTE -total,
                  data=nozero_manual_grams_3outcomes_train,
                  method = "svmRadial",
                  metric="Accuracy",
                  tuneGrid = grid,
                  trControl=ctrl,
                  verbose=T)

plot(svm.tune2)

test_results4 <- predict(svm.tune2, nozero_manual_grams_3outcomes_test)
confusionMatrix(test_results4, nozero_manual_grams_3outcomes_test$TBQ)

grid2 <- expand.grid(scale = c(seq(0.1,0.5, by=0.05)),
                    C = c(7.50, 8, 8.5, 9))

svm.tune3 <- train(TBQ~. -notbqdata -ID_PACIENTE -total,
                   data=nozero_manual_grams_3outcomes_train,
                   method = "svmPoly",
                   metric="Accuracy",
                   tuneLength= 3,
                   trControl=ctrl,
                   verbose=T)
