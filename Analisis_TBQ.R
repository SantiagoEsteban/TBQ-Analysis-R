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
shortcorpus <- Corpus(VectorSource(corptext[1:10]))

a
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
corpus2 <- tm_map(corpus2, toSpace, " l ")
corpus2 <- tm_map(corpus2, removePunctuation)
corpus2 <- tm_map(corpus2, content_transformer(tolower))
corpus2 <- tm_map(corpus2, removeNumbers)
corpus2 <- tm_map(corpus2, stripWhitespace)

shortcorpus2 <- tm_map(shortcorpus, toSpace, ":")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, "\\.")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, "_x000D_\r\n")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, ",")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, ";")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, "\\?")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, "!")
shortcorpus2 <- tm_map(shortcorpus2, toSpace, " l ")
shortcorpus2 <- tm_map(shortcorpus2, removePunctuation)
shortcorpus2 <- tm_map(shortcorpus2, content_transformer(tolower))
shortcorpus2 <- tm_map(shortcorpus2, removeNumbers)
shortcorpus2 <- tm_map(shortcorpus2, stripWhitespace)

#Analysis with quanteda
#Steps: 1) tokenize 
#       2) remove all non related words(will get this from sas variables created manually)
#       3) ngrams 1:3
#       4) Stem?
#       5) DocumentTermMatrix (see below)
#       5) Analyze according to A gentle introduction to text mining using R and RTextTools: A Supervised Learning Package for Text Classification

qcorpus <- corpus(corpus2) #quanteda corpus
docvars(qcorpus, "TBQ") <- corp$TBQ
docvars(qcorpus, "FECHA") <- corp$FECHA
metadoc(qcorpus, "language") <- "es"
summary(qcorpus)

#Crear la list de palabaras de remover, osea todas las que no sean TBQ TERMS
tbqterms <- as.character(read.csv('Terminos TBQ.txt', header=FALSE)$V1) #TBQ TERMS
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

#DFM
tbqtermsstopwords <- as.character(read.csv('Terminos TBQ - a ignorar.txt', header=FALSE)$V1) #TBQ TERMS STOPWORDS
dfmtokenqcorpus <- dfm(ngramcleantokenqcorpus, verbose=TRUE, language='spanish', 
                       ignoredFeatures = tbqtermsstopwords)
#Ya podria pasar a tm o a RtexTools
# HASTA ACA ESTA BIEN




tokenqcorpus <- tokenize(texts(qcorpus), removeHyphens=TRUE, simplify=FALSE, verbose=TRUE)
tokenqcorpus2 <- corpus(tokenqcorpus)
cleantokenqcorpus <- selectFeatures(texts(tokenqcorpus), tbqterms, selection='keep', 
                                    valuetype='fixed', case_insensitive=TRUE, verbose=TRUE)
dfmtokenqcorpus <- dfm(tokenqcorpus, verbose=TRUE, keptFeatures = tbqterms, language='spanish',
                       valuetype='fixed')

##VER ACA, ESTOY INTENTANDO REARMAR EL CORPUS, CREO Q VOY BIEN.











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
