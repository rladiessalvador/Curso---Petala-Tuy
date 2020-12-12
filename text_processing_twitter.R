##############################################
############## Minerando dados ###############
##############################################


# Instalando e carregando pacotes necessários
installedPackages = rownames(installed.packages())

ifelse('twitteR' %in% installedPackages == FALSE,install.packages('twitteR'),library('twitteR'))
ifelse('wordcloud' %in% installedPackages == FALSE,install.packages('wordcloud'),library('wordcloud'))
ifelse('data.table' %in% installedPackages == FALSE,install.packages('data.table'),library('data.table'))
rm(installedPackages)

# Registrando sessão do R no Twitter
apiKey = '******************'
apiSecretKey = '*****************'
accessToken = '******************'
accessSecret = '******************'

setup_twitter_oauth(apiKey, apiSecretKey,accessToken, accessSecret)
rm(apiKey,apiSecretKey,accessToken,accessSecret)

# Minerando posts no twitter
search <- searchTwitter('covid -filter:links', 
                        n = 20000,
                        lang = 'pt',
                        since = '2020-03-01')

# Transformando posts minerados em data table
search <- as.data.table(twListToDF(search))

# Definindo colunas de interesse
df <- search[,.(id,screenName,text,created,favoriteCount,retweetCount,isRetweet)]
rm(search)
# Eliminando tweets repetidos
df <- unique(df,by = 'id')
View(df[1:200,])

#Salvando dados processados
save(df,file = "TwitterCovid.RData")


##############################################
############## Pré processamento #############
##############################################

library(tm); library(ggplot2)

load("TwitterCovid.RData")

clean_corpus<-function(text){
  text <- gsub('\\p{So}|\\p{Cn}', '',text, perl = TRUE)   #Remove emojis
  corp <- Corpus(VectorSource(text)) # Creatign corpus
  corp <- tm_map(corp, content_transformer(function(x) gsub("htt[[:alnum:][:punct:]]*|www[[:alnum:][:punct:]]*|[^[:alnum:]///'#% ]", " ", x))) #Remove URL and LINK
  corp <- tm_map(corp, tolower) # Transforming content to lowercase
  corp <- tm_map(corp, PlainTextDocument) 
  corp <- tm_map(corp, removeNumbers)  #Removing numbers
  swList <- c(stopwords('portuguese'),'rt') # Incluindo 'rt' como uma stopword
  corp <- tm_map(corp, removeWords, swList) # Removing stopwords
  corp <- tm_map(corp, stemDocument, "portuguese") #Stemming 
  corp <- tm_map(corp, stripWhitespace) # Removendo espaços em branco 
}

df$clean_text <- data.frame(text = sapply(clean_corpus(df$text), as.character)$content, stringsAsFactors = FALSE)
View(df[1:100,.(text,clean_text)])

# Construindo uma matriz de termos e documentos com pesos TF-IDF
contrl = list(weighting = function(x) weightTfIdf(x))
TDM <- TermDocumentMatrix(Corpus(VectorSource(df$clean_text)),control = list(weighting = weightTfIdf))
inspect(TDM[0:200,0:200])

# Construindo nuvem de palavras
matrix <- as.matrix(TDM) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
head(words,10)
wordsfreq <- data.frame(word = names(words),freq=words)
head(wordsfreq)
wordcloud2(data=wordsfreq, size=1.6, color='random-dark')

#############################################################################
###################### Aprendizado de máquina ###############################
#############################################################################

# Criando classificação dos tweets sengundo o sentimento positivo e negativo
sentimentos <- c('Positivo','Negativo')
lista_sentimentos <- list()
lista_sentimentos[['Positivo']] <- c('esperança','alegria','graças deus','venci ','alívio','felicidade','curei','começou vacinar','avanços',' cura ')
lista_sentimentos[['Negativo']] <- c('morte','horrível','irrita','tenso','odeio','depressão','mata')

for (i in 1:length(sentimentos)){
  print(paste0('Identificando palavras do sentimento: ',sentimentos[i]))  
  palavras <- lista_sentimentos[[sentimentos[i]]]
  for (j in 1:length(palavras)){
    print(palavras[j])
    df[grep(palavras[j], clean_text), sentimento := sentimentos[i]]
  }
}; rm(i,j,lista_sentimentos,palavras,sentimentos)

table(df$sentimento)
View(df[,.(id,text,clean_text,sentimento)])

# Balanceando conjunto de dados
ids_negativos <- sample(df[sentimento %in% 'Negativo']$id,216,replace = F)
df_sentimento <- rbind(df[id %in% ids_negativos],df[sentimento %in% 'Positivo']); rm(ids_negativos)
table(df_sentimento$sentimento)
View(df_sentimento[,.(id,text,clean_text,sentimento)])


##############################################
############## Modelagem #####################
##############################################

library(RTextTools)

# Definindo conjunto de teste e treino
df_sentimento$train <- ifelse(runif(nrow(df_sentimento))<0.80,1,0)

#Embaralhando dataset
set.seed(42)
rows <- sample(nrow(df_sentimento))
df_sentimento <- df_sentimento[rows, ]


# Treinando o modelo
matrixtrain <- create_matrix(df_sentimento$clean_text)#; rm(corp,text)

train_size <- 1:sum(df_sentimento$train)
test_size <- (sum(df_sentimento$train)+1):nrow(df_sentimento)

container = create_container(matrixtrain, 
                             as.numeric(as.factor(df_sentimento$sentimento)),
                             trainSize=train_size, 
                             testSize=test_size,
                             virgin=FALSE)

models = train_models(container, algorithms=c("MAXENT" , "SVM", "RF", "BAGGING"))

# Validando os resultados
results = classify_models(container, models)
analyticsdb <- create_analytics(container,results)

fscores <- as.data.table(cbind(attributes(colMeans(analyticsdb@algorithm_summary[,grepl("FSCORE",names(analyticsdb@algorithm_summary))]))$names,colMeans(analyticsdb@algorithm_summary[,grepl("FSCORE",names(analyticsdb@algorithm_summary))])))

table(as.numeric(as.factor(df_sentimento[test_size]$sentimento)), results[,"FORESTS_LABEL"])
table(as.numeric(as.factor(df_sentimento[test_size]$sentimento)), results[,"SVM_LABEL"])
summary(analyticsdb)


# Usando modelo para predizer novos tweets

prediction_data <- list(df[id %in% '1336456904667697154']$clean_text[1],
                        df[id %in% '1336456788783337472']$clean_text[1],
                        df[id %in% '1336456575536521216']$clean_text[1])

predictionmatrix <- create_matrix(prediction_data, 
                                  language='portuguese',
                                  originalMatrix=matrixtrain)
predictioncontainer <- create_container(predictionmatrix,
                              c(rep(NA,length(prediction_data))),
                              testSize=1:length(prediction_data),
                              virgin=T)

results <- classify_models(predictioncontainer, models)
