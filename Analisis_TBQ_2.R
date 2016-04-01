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
cl<-makeCluster(3) #change the 4 to your number of CPU cores
registerDoSNOW(cl)
#library(doParallel)

##################
#Import manual dataset created with REGEX SAS
##################

manual_grams <- as.data.frame(read_excel('tbqcompleto2015_FINAL_nodups2.xlsx'))
manual_grams <- select(manual_grams, -notbqdata, -total)

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
