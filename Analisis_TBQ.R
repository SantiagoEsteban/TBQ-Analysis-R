# Import data
#A gentle introduction to text mining using R
library(readxl)
library(tm)
library(RTextTools)
library(NLP)
library(kernlab)
library(quanteda)
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


#FOR N-GRAM VERSION 
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
library(RWeka)
FourgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 4))
Four_gram_tdm <- DocumentTermMatrix(corpus2, control=list(tokenize=FourgramTokenizer, 
                                                weighting=function(x) weightTfIdf(x, normalize =FALSE)))

#Tidy up Four_gram_tdm
#Removing sparse terms
Four_gram_tdm
Four_gram_tdm_nosparse <- removeSparseTerms(Four_gram_tdm, 0.989)

#removing zero variance terms
library(caret)
library(tidyr)
library(dplyr)
Four_gram_tdm_df <- tbl_df(as.data.frame(as.matrix(Four_gram_tdm_nosparse)))
nearZeroVar(Four_gram_tdm_df) #too many

nozero_Four_gram_tdm_df <- colSums(Four_gram_tdm_df) > 0
Four_gram_tdm_df <- Four_gram_tdm_df[, nozero_Four_gram_tdm_df]

#removing highly correlated
library(corrplot)
par(mfrow=c(1,1))
Four_gram_tdm_df_cor <- cor(Four_gram_tdm_df)
corrplot(Four_gram_tdm_df_cor, order = "hclust", tl.cex=0.2)
#Finding highly correlated variables
highCorr <- findCorrelation(Four_gram_tdm_df_cor, cutoff = .75)
highCorr
Four_gram_tdm_df_nocor <- Four_gram_tdm_df[, -highCorr]
corrplot(cor(Four_gram_tdm_df_nocor), order = "hclust", tl.cex=0.2)

#LISTO HASTA ACA N-GRAMS


##############################
#Generar tdm solo con los terminos sueltos, sin n-grams
##############################
one_gram_tdm_tf <- DocumentTermMatrix(corpus2, control=list(wordLengths=c(1, Inf), dictionary=tbqterms))
one_gram_tdm_tfidf <- DocumentTermMatrix(corpus2, control=list(wordLengths=c(1, Inf), dictionary=tbqterms, weighting=function(x) weightTfIdf(x, normalize =FALSE)))

#Analyzing frequencies
one_gram_tdm_tf_df <- tbl_df(as.data.frame(as.matrix(one_gram_tdm_tf)))
one_gram_tdm_tfidf_df <- tbl_df(as.data.frame(as.matrix(one_gram_tdm_tfidf)))
freq_one_gram_tdm_tf_df <- colSums(one_gram_tdm_tf_df)
nozero_freq_one_gram_tdm_tf_df <- freq_one_gram_tdm_tf_df > 0


freq_one_gram_tdm_tfidf_df <- colSums(one_gram_tdm_tfidf_df)
freq_one_gram_tdm_tfidf_df
order(freq_one_gram_tdm_tf_df, decreasing=TRUE)
order(freq_one_gram_tdm_tfidf_df, decreasing=TRUE)

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
corrplot(manual_grams_cor, order = "hclust", tl.cex=0.5)
#Finding highly correlated variables
findCorrelation(manual_grams_cor, cutoff = .75) #no correlations above 0.75

###############
#DONE manual_grams
###############



library(caret)
tdmdataframe <- as.data.frame(as.matrix(tdm))
tdmdataframenzv <- preProcess(tdmdataframe, method="nzv")
#Creating the DOC TERM MATRIX in tm, with the dictionary
#Al aplciar el diccionario solo quedan los tokens solos, no quedan los ngrams
# podria probar asi con los 124 tokens y usando modelos no lineales a ver como funciona para ver la interacción).
# Sino mas arriba esta el codigo para volar todos los stopwords no tbq y armar los ngram con eso
# Despues hay que limpiar por que queda lleno de no_no_no de_de_de y cosas asi.
# En definitiva son *** formas de generar las variables: 
#1. la 100% manual (lo que hice en SAS)
#2. solo con las variable del diccionario (sin ngrams)
#3. los ngrams usando stopwords (el diccionario de palabras no tbq), 
#4. Tendria que probar si puedo aplciar el diccionario sobre los ngrams pero usarlo como regex por ejemplo o como glo
# en vez de fixed y ver si eso ahce que se queden los ngrams en vez de los terminos sueltos del diccionario
# 5. Usar todo el corpus para armar la dtm con ngrams, despues seleccionar variables usando el 
# diccionario (selectfeatures creo q era) y ver que queda.

#Todas las opciones deben pasar por un proceso de limpiez estadistico (nzerovar, removesparseterms, etc)

install.packages('RWeka')
library(RWeka)

dicttbq <- Corpus(VectorSource(tbqterms))

#Create four ngram tokenizer function
corpus3 <- corpus(corptext, dictionary=tbqterms)
FourgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 4))
tdm <- DocumentTermMatrix(corpus2, control=list(dictionary=tbqterms))

topfeatures(dfmtokenqcorpus2, verbose=FALSE)

#Modeling


#Ya podria pasar a tm o a RtexTools
# HASTA ACA ESTA BIEN




tokenqcorpus <- tokenize(texts(qcorpus), removeHyphens=TRUE, simplify=FALSE, verbose=TRUE)
tokenqcorpus2 <- corpus(tokenqcorpus)
cleantokenqcorpus <- selectFeatures(texts(tokenqcorpus), tbqterms, selection='keep', 
                                    valuetype='fixed', case_insensitive=TRUE, verbose=TRUE)
dfmtokenqcorpus <- dfm(tokenqcorpus, verbose=TRUE, keptFeatures = tbqterms, language='spanish',
                       valuetype='fixed')

##VER ACA, ESTOY INTENTANDO REARMAR EL CORPUS, CREO Q VOY BIEN.






##########################################################################

qcorpus <- corpus(corpus2) #quanteda corpus
docvars(qcorpus, "TBQ") <- corp$TBQ
docvars(qcorpus, "FECHA") <- corp$FECHA
metadoc(qcorpus, "language") <- "es"
summary(qcorpus)

#Crear la list de palabaras de remover, osea todas las que no sean TBQ TERMS
tbqterms <- unique(as.character(read.csv('Terminos TBQ.txt', header=FALSE)$V1)) #TBQ TERMS
library(plyr)
'%nin%' <- Negate('%in%')
nontbqterms <- lapply(texts(qcorpus), function(x) {
    t <- unlist(strsplit(x, " "))
    t[t %nin% tbqterms]
})
nontbqterms2 <- unlist(nontbqterms)


#Tokenization
tokenqcorpus <- tokenize(texts(qcorpus), simplify=FALSE, removeHyphens=TRUE, verbose=TRUE)
shortqcorpus <- corpus(shortcorpus2)
shorttokenqcorpus <- tokenize(shortqcorpus)

#Removing non TBQ terms - Funciona en el mas corto, hay q darle tiempo al largo o cortarlo
cleantokenqcorpus <- removeFeatures(tokenqcorpus, stopwords=nontbqterms2, verbose=TRUE)
cleanshorttokenqcorpus <- removeFeatures(shorttokenqcorpus, stopwords=nontbqterms2, verbose=TRUE)

#Ngrams
ngramcleantokenqcorpus <- ngrams(cleantokenqcorpus, 1:4)
ngramstokenqcorpus <- quanteda::ngrams(tokenqcorpus, 1:4)

#DFM
tbqtermsstopwords <- as.character(read.csv('Terminos TBQ - a ignorar.txt', header=FALSE)$V1) #TBQ TERMS STOPWORDS
dfmtokenqcorpus <- dfm(ngramcleantokenqcorpus, verbose=TRUE, language='spanish', 
                       ignoredFeatures = tbqtermsstopwords) #DocTerm - Matrix in QUANTEDA


#Esta DTM queda con 124 features que son las del diccionario, no se queda con los ngrams, solo monograms.
tbqterms <- unique(as.character(read.csv('Terminos TBQ.txt', header=FALSE)$V1)) #TBQ TERMS
names(tbqterms) <- tbqterms
qdictio <- dictionary(as.list(tbqtermsglob))
dfmtokenqcorpus2 <- dfm(ngramstokenqcorpus, verbose=TRUE, language='spanish', 
                        dictionary=qdictio, valuetype = 'glob') #DocTerm - Matrix in QUANTEDA


corpus2 <- tm_map(corpus2, stemDocument(corpus2, language= meta(corpus2, "spanish")))
#no funciona en todo el corpus, pro si de a uno, se podria hacer algun loop. 
#Igual no se si vale la pena. Por ejemplo fuma lo pasa a fum...

corpus2 <- tm_map(corpus2, removeWords, stopwords("spanish"))
#este funciona pero borro "no" por ejemplo... habria que ver si se peude acceder a la lista

#Clustering
corpuslist <- as.list(corpus2)
corpuslistshort <- corpuslist[1:100]
stringkern <- stringdot(length=4, type = "string")
stringCl <- specc(corpuslistshort, 3, kernel = stringkern)
table("String Kernel" = stringCl, corp$TBQ[1:100])

#Create DocumentTermMatrix
dtm <- DocumentTermMatrix(corpus2)
inspect(dtm[1:2,1000:1005])
freq <- colSums(as.matrix(dtm))
length(freq)
ord <- order(freq,decreasing=TRUE)
freq[head(ord)]
freq[tail(ord)]

#Limiting the DTM
dtmr <-DocumentTermMatrix(corpus2, control=list(wordLengths=c(4, 20),
                                                bounds = list(global = c(3,27))))
freq2 <- colSums(as.matrix(dtmr))
length(freq2)
ord <- order(freq2,decreasing=TRUE)
freq2[head(ord)]
freq2[tail(ord)]

#RTextTools: A Supervised Learning Package for Text Classification
texts <-read_excel('tbqcompleto2015_validar3_nodups1.xlsx')

#PODRIA PROBAR DE USAR LA FUNCION toSpace antes de trnasformarlo en docmatrix.

texts <- sapply(texts$TEXTO, toSpace("_x000D_\r\n"))

doc_matrix <- create_matrix(ngramcleantokenqcorpus)
