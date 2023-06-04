library(caret)
library(e1071)
library(quanteda)
library(irlba)
library(dplyr)
library(ggplot2)
library(randomForest)

raw_spam_data<-read.csv(file = file.choose(), stringsAsFactors = F)
View(raw_spam_data)

raw_spam_data<-raw_spam_data[,1:2] #Reading only the first two columns
names(raw_spam_data)<-c("Label","Text") #Naming the two columns
raw_spam_data$Label<-as.factor(raw_spam_data$Label) #turning Label columns in to factor of two #categories
levels(raw_spam_data$Label)
length(which(complete.cases(raw_spam_data))) #checking if all the rows don't have any missing values
prop.table(table(raw_spam_data$Label))

raw_spam_data$TextLength<-nchar(raw_spam_data$Text)

tokens<-tokens(raw_spam_data$Text, what = "word", remove_punct = T, remove_numbers = T,
                     remove_symbols = T, split_hyphens = T)
tokens<-tokens_tolower(tokens) #converts all token words in lower form
tokens<-tokens_select(tokens, stopwords(), selection = "remove" )#removes all the 
#stop words from dictionary
tokens<-tokens_wordstem(tokens, language = "english")

tokens[[1]]

tokens_dfm<-dfm(tokens, tolower = F)
tokens_matrix<-as.matrix(tokens_dfm)
tokens_matrix[1:15,1:15]


TF.function<-function(r){
  r/sum(r)
  } #This will take the frequency of a particular word in a document and divide by the total 
  #word count in that document

IDF.function<-function(c){
  size<-length(c) #Count the total number of documents
  doc.count<-length(which(c>0)) # For a certain word, count the number of documents where it appeared
  log10(size/doc.count)
}

tf.idf<-function(tf,idf){
  tf*idf
}


#different tf-idf function
TF.function<-function(r){
  r/sum(r)
} #This will take the frequency of a particular word in a document and divide by the total 
#word count in that document

IDF.function<-function(c){
  size<-length(c) #Count the total number of documents
  doc.count<-length(which(c>0)) # For a certain word, count the number of documents where it appeared
  asinh(size/doc.count)
}

tf.idf<-function(tf,idf){
  tf*idf
}


token.tf<-apply(tokens_matrix,1,TF.function) # '1' has been used for row operation
token.tf[1:10,1:10]
token.idf<-apply(tokens_matrix,2,IDF.function)

token.tf_idf<-apply(token.tf,2,tf.idf,idf=token.idf)
token.tf_idf[1:14,1:9]
token.tf_idf<-t(token.tf_idf)

incomplete.case.index<-which(!complete.cases(token.tf_idf))
raw_spam_data$Text[incomplete.case.index]

token.tf_idf[incomplete.case.index,]<-rep(0.0,ncol(token.tf_idf))
sum(which(!complete.cases(token.tf_idf)))

dim(token.tf_idf)

token.tf_idf.df<-cbind(raw_spam_data$Label, data.frame(token.tf_idf))
names(token.tf_idf.df)<-make.names(names = names(token.tf_idf.df))
colnames(token.tf_idf.df)[1]<-"Label"

set.seed(32984, sample.kind = "Rounding")
index<-createDataPartition(token.tf_idf.df$Label, times = 1, p=0.7, list = F) #create data 
#partition function will always maintain the correct proportion of ham and spam in the train and test data.

test_set<-token.tf_idf.df[-index,]
train_set<-token.tf_idf.df[index,]

prop.table(table(test_set$Label))
prop.table(table(train_set$Label))

cv_folds<-createMultiFolds(train_set$Label, k = 10,times = 3) #creating 30 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv_folds)

library(doSNOW) #This library is required for multicore processing
start_time<-Sys.time()
clusters<-makeCluster(2, type = "SOCK") #This will instruct rstudio to use two cores simultaneously 
registerDoSNOW(clusters) #Clustering will begin
trained_model_01<-train(Label~., data = train_set, method="rpart", trControl=train_control,
                        tuneLength=7) #Training a decision tree model
stopCluster(clusters) #Ending the clustering and stopping multicore processing
total_time<-Sys.time()-start_time


trained_model_01

predictions<-predict(trained_model_01, newdata = test_set[,-1], type = "raw")
confusionMatrix(test_set$Label, predictions)

library(e1071)
svm.fit<-svm(Label ~ ., data = train_set, kernel = "linear", cost = 10, scale = FALSE)

predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Label)


library(irlba)
start.time<-Sys.time()
projected.data<-irlba(t(token.tf_idf), nv = 300, maxit = 600)
total.time<-Sys.time()-start.time
total.time

projected.data$v[1:10,1:6]

sigma.inverse<-1/projected.data$d
u.transposed<-t(projected.data$u)
doc.01<-token.tf_idf[1,]
doc01.inverse.trans<-sigma.inverse * u.transposed %*% doc.01

projected.data$v[1,1:10]
doc01.inverse.trans[1:10]

svd.data<-data.frame(Labels=token.tf_idf.df$Label, projected.data$v)
dim(svd.data)

test_set<-svd.data[-index,]
train_set<-svd.data[index,]

cv_folds<-createMultiFolds(train_set$Labels, k = 10,times = 3) #creating 30 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv_folds)

start_time<-Sys.time()

trained_model.svd<-train(Labels ~ ., data = train_set, method="rpart",
                           trControl=train_control, tuneLength=7)

total_time<-Sys.time()-start_time
total_time

predictions<-predict(trained_model.svd, test_set[,-1])
confusionMatrix(predictions, test_set$Labels)


svm.fit<-svm(Labels ~ ., data = train_set, kernel = "linear", cost = 10, scale = FALSE)

predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Labels)


#BERT on SPAM dataset
library(readr)
library(keras)
library(tensorflow)
library(stringi)
library(reticulate)
library(tm)
library(stringr)

raw_spam_data<-read.csv(file = file.choose(), stringsAsFactors = F)

head(raw_spam_data)
dim(raw_spam_data)

raw_spam_data<-raw_spam_data[,1:2] #Reading only the first two columns
names(raw_spam_data)<-c("Label","Text")
View(raw_spam_data)

Labels<-raw_spam_data$Label
Labels01<-ifelse(Labels=="ham",1,0)
Labels[1:10]
Labels01[1:10]

Labels01.encoded<-to_categorical(Labels01)

transformer = reticulate::import('transformers')
tf = reticulate::import('tensorflow')
builtins <- import_builtins() #built in python methods

multilingual.tokenizer <- transformer$AutoTokenizer$from_pretrained('bert-base-uncased')
multilingual.BERT = transformer$TFBertModel$from_pretrained("bert-base-uncased")

n=length(raw_spam_data$Text)
n
features_train = matrix(NA, nrow=n, ncol=768)
dim(features_train)


for (i in 1:n){
  encodings_i = multilingual.tokenizer(raw_spam_data$Text[i], 
                                       truncation=TRUE, padding=TRUE,max_length=250L, return_tensors='tf')
  features_train[i,] = py_to_r(array_reshape(multilingual.BERT(encodings_i)[[1]][[0]][[0]],c(1, 768)))
}


dim(features_train)
saveRDS(object = features_train, file = "feature trained spam data.rds")


model <- keras_model_sequential() %>% 
  # Specify the input shape
  layer_dense(units = 100, activation = "relu", input_shape = ncol(features_train)) %>% 
  # add a dense layer with 40 units
  layer_dense(units = 40, activation = "relu", kernel_initializer = "he_normal", 
              bias_initializer = "zeros", kernel_regularizer = regularizer_l2(0.05)) %>% 
  layer_dropout(rate = 0.2) %>%
  # add the classifier on top
  layer_dense(units = 2, activation = "sigmoid")

model %>% compile(
  optimizer="rmsprop",
  loss="binary_crossentropy",
  metrics= c("accuracy")
)

history <- model %>% fit(
  features_train, Labels01.encoded,
  epochs = 20,
  batch_size = 32,
  validation_split=0.3)



#Deep Learning Keras modeling for IMDB dataset
library(readr)
library(keras)
library(tensorflow)
library(stringi)
library(reticulate)
library(tm)
library(stringr)

imdb.data<-read.csv(file = file.choose(), stringsAsFactors = FALSE)
imdb.data[1:3,]


index<-sample(1:50000)
imdb.data<-imdb.data[index[1:25000],]
dim(imdb.data)

#removing punctuation marks from the reviews
imdb.data$review<-gsub('[[:punct:] ]+',' ', imdb.data$review)

#removing the stopwords
stopwords<- paste(stopwords('en'), collapse = '\\b|\\b')
stopwords<- paste0('\\b', stopwords, '\\b')
imdb.data$review<- stringr::str_replace_all(imdb.data$review, stopwords, '')

#creating the labels

Labels<-imdb.data$sentiment
levels(as.factor(Labels))
Labels<-ifelse(Labels=="positive",1,0)
saveRDS(object = Labels, file = "imdb labels 25000.rds")
Labels01.encoded<-to_categorical(Labels)
class(Labels)

#Tokenizing the reviews

tokenizer<- text_tokenizer(num_words = 10000) %>% fit_text_tokenizer(imdb.data$review)

# and put these integers into a sequence
sequences <- texts_to_sequences(tokenizer, imdb.data$review)
class(sequences)

# and make sure that every sequence has the same length (Keras requirement)
data <- pad_sequences(sequences, maxlen = 150)
dim(data)

#vectorizing the sequences
vectorize.sequences<-function(sequences, dimension=10000){
  results<-matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences)){
    results[i, sequences[[i]]]<-1
  }
  results
}

train.data<-vectorize.sequences(sequences)
rm(sequences)
rm(tokenizer)
rm(imdb.data)

#creating training data and validation data
val.indices<-1:10000
x.val<-train.data[val.indices,]
partial.x.train<-train.data[-val.indices,]
y.val<-Labels[val.indices]
partial.y.train<-Labels[-val.indices]


dim(x.val)
dim(partial.x.train)

#keras modelling
model<-keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer="rmsprop",
  loss="binary_crossentropy",
  metrics= c("accuracy")
)

#model fitting
history <- model %>% fit(
  partial.x.train,
  partial.y.train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x.val, y.val)
)



#BERT for IMDB data Sentiment analysis

transformer = reticulate::import('transformers')
tf = reticulate::import('tensorflow')
builtins <- import_builtins() #built in python methods

multilingual.tokenizer <- transformer$AutoTokenizer$from_pretrained('bert-base-uncased')



text = "I love you"
multilingual.tokenizer(text)

text2<-imdb.data$review[10]
text2

example.tokenize<-multilingual.tokenizer$encode(text2)
example.tokenize
multilingual.tokenizer$convert_ids_to_tokens(example.tokenize)
multilingual.tokenizer$decode(example.tokenize)

multilingual.BERT = transformer$TFBertModel$from_pretrained("bert-base-uncased")

n=length(imdb.data$review)
n
features_train = matrix(NA, nrow=n, ncol=768)
dim(features_train)


for (i in 1:n){
  encodings_i = multilingual.tokenizer(imdb.data$review[i], 
                                       truncation=TRUE, padding=TRUE,max_length=250L, return_tensors='tf')
  features_train[i,] = py_to_r(array_reshape(multilingual.BERT(encodings_i)[[1]][[0]][[0]],c(1, 768)))
}


dim(features_train)
saveRDS(object = features_train, file = "feature trained IMDB.rds")


model <- keras_model_sequential() %>% 
  # Specify the input shape
  layer_dense(units = 100, activation = "relu", input_shape = ncol(features_train)) %>% 
  # add a dense layer with 40 units
  layer_dense(units = 40, activation = "relu", kernel_initializer = "he_normal", 
              bias_initializer = "zeros", kernel_regularizer = regularizer_l2(0.05)) %>% 
  layer_dropout(rate = 0.2) %>%
  # add the classifier on top
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer="rmsprop",
  loss="binary_crossentropy",
  metrics= c("accuracy")
)

history <- model %>% fit(
  features_train, Labels,
  epochs = 20,
  batch_size = 32,
  validation_split=0.3)


model <- keras_model_sequential() %>% 
  # Specify the input shape
  layer_dense(units = 64, activation = "relu", input_shape = ncol(features_train)) %>% 
  # add a dense layer with 40 units
  layer_dense(units = 32, activation = "relu") %>%
  # add the classifier on top
  layer_dense(units = 1, activation = "sigmoid") 


model %>% compile(
  optimizer="rmsprop",
  loss="binary_crossentropy",
  metrics= c("accuracy")
)


history <- model %>% fit(
  features_train, Labels,
  epochs = 20,
  batch_size = 32,
  validation_split=0.3)



