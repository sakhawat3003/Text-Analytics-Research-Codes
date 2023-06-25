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

#tokenization
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


#TF-IDF function
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


#CTF-IDF function
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

#applying tf function
token.tf<-apply(tokens_matrix,1,TF.function) # '1' has been used for row operation
token.tf[1:10,1:10]
#applying idf function
token.idf<-apply(tokens_matrix,2,IDF.function)

#applying tf-idf function
token.tf_idf<-apply(token.tf,2,tf.idf,idf=token.idf)
token.tf_idf[1:14,1:9]
#application of the tf-idf function puts the features in rows so it's required to transpose 
token.tf_idf<-t(token.tf_idf)

incomplete.case.index<-which(!complete.cases(token.tf_idf))
raw_spam_data$Text[incomplete.case.index]

#replacing all the values in empty rows with zeroes
token.tf_idf[incomplete.case.index,]<-rep(0.0,ncol(token.tf_idf))
sum(which(!complete.cases(token.tf_idf)))

dim(token.tf_idf)

#merging the Label column with the tf-idf transformed data
token.tf_idf.df<-cbind(raw_spam_data$Label, data.frame(token.tf_idf))
names(token.tf_idf.df)<-make.names(names = names(token.tf_idf.df))
colnames(token.tf_idf.df)[1]<-"Label"

#partitioning the dataset in 70:30 ratio
set.seed(32984, sample.kind = "Rounding")
index<-createDataPartition(token.tf_idf.df$Label, times = 1, p=0.7, list = F) #create data 
#partition function will always maintain the correct proportion of ham and spam in the train and test data.

test_set<-token.tf_idf.df[-index,]
train_set<-token.tf_idf.df[index,]

prop.table(table(test_set$Label))
prop.table(table(train_set$Label))

#creating 10 folds cross validation
cv_folds<-createMultiFolds(train_set$Label, k = 10,times = 1) #creating 30 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 1, index = cv_folds)

#training the decision tree model
library(doSNOW) #This library is required for multicore processing
start_time<-Sys.time()
clusters<-makeCluster(2, type = "SOCK") #This will instruct rstudio to use two cores simultaneously 
registerDoSNOW(clusters) #Clustering will begin
trained_model_01<-train(Label~., data = train_set, method="rpart", trControl=train_control,
                        tuneLength=7) #Training a decision tree model
stopCluster(clusters) #Ending the clustering and stopping multicore processing
total_time<-Sys.time()-start_time


trained_model_01

#prediction with the decision tree model
predictions<-predict(trained_model_01, newdata = test_set[,-1], type = "raw")
confusionMatrix(test_set$Label, predictions)

#training the support vector machine model
library(e1071)
svm.fit<-svm(Label ~ ., data = train_set, kernel = "linear", cost = 10, scale = FALSE)

#prediction with the support vector machine model
predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Label)

#transformation with the IRLBA
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

#merging the Label column with IRLBA transformed data
svd.data<-data.frame(Labels=token.tf_idf.df$Label, projected.data$v)
dim(svd.data)

test_set<-svd.data[-index,]
train_set<-svd.data[index,]

#creating 10 fold cross validation 
cv_folds<-createMultiFolds(train_set$Labels, k = 10,times = 1) #creating 30 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 1, index = cv_folds)

start_time<-Sys.time()

#training the decision tree model for IRLBA transformed data
trained_model.svd<-train(Labels ~ ., data = train_set, method="rpart",
                           trControl=train_control, tuneLength=7)

total_time<-Sys.time()-start_time
total_time

#prediction with the decision tree model
predictions<-predict(trained_model.svd, test_set[,-1])
confusionMatrix(predictions, test_set$Labels)

#training the support vector machine for IRLBA transformed data
svm.fit<-svm(Labels ~ ., data = train_set, kernel = "linear", cost = 10, scale = FALSE)

#prediction with the support vector machine
predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Labels)


#Deep learning with Keras and BERT on SPAM dataset
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

#importing "transformer" and "tensorflow"
transformer = reticulate::import('transformers')
tf = reticulate::import('tensorflow')
builtins <- import_builtins() #built in python methods

multilingual.tokenizer <- transformer$AutoTokenizer$from_pretrained('bert-base-uncased')
multilingual.BERT = transformer$TFBertModel$from_pretrained("bert-base-uncased")

#training the BERT model with mBERT
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

#creating deep learning Keras API
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



#application of the methodology on SMS Phishing dataset
library(caret)
library(e1071)
library(quanteda)
library(irlba)
library(dplyr)
library(ggplot2)
library(randomForest)

raw_spam_data<-read.csv(file = file.choose(), stringsAsFactors = F)#load SMS Phishing Data
View(raw_spam_data)

raw_spam_data<-raw_spam_data[,1:2] #Reading only the first two columns
names(raw_spam_data)<-c("Label","Text") #Naming the two columns

raw_spam_data$Label<-ifelse(raw_spam_data$Label %in% c("smishing","Smishing","Spam","spam") ,"spam","ham")

raw_spam_data$Label<-as.factor(raw_spam_data$Label) #turning Label columns in to factor of two categories
levels(raw_spam_data$Label)


length(which(complete.cases(raw_spam_data))) #checking the number of rows with missing values
prop.table(table(raw_spam_data$Label))

#Tokenization
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

#traditional Tf-Idf function
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


#CTf-Idf function

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

#applying tf function
token.tf<-apply(tokens_matrix,1,TF.function) # '1' has been used for row operation
token.tf[1:10,1:10]
#applying idf function
token.idf<-apply(tokens_matrix,2,IDF.function)

#applying ctf-idf function
token.tf_idf<-apply(token.tf,2,tf.idf,idf=token.idf)
token.tf_idf[1:14,1:9]

#application of ctf-idf puts the features in rows. So it's required to transpose it.  
token.tf_idf<-t(token.tf_idf)

incomplete.case.index<-which(!complete.cases(token.tf_idf))
raw_spam_data$Text[incomplete.case.index]

#replacing all the rows with NA values with zeroes
token.tf_idf[incomplete.case.index,]<-rep(0.0,ncol(token.tf_idf))
sum(which(!complete.cases(token.tf_idf)))

dim(token.tf_idf)

#merging the Label column with the tf-idf transformed data 
token.tf_idf.df<-cbind(raw_spam_data$Label, data.frame(token.tf_idf))
names(token.tf_idf.df)<-make.names(names = names(token.tf_idf.df))
colnames(token.tf_idf.df)[1]<-"Label"

#partitioning the data
set.seed(32984, sample.kind = "Rounding")
index<-createDataPartition(token.tf_idf.df$Label, times = 1, p=0.7, list = F) #create data 
#partition function will always maintain the correct proportion of ham and spam in the train and test data.

test_set<-token.tf_idf.df[-index,]
train_set<-token.tf_idf.df[index,]

prop.table(table(test_set$Label))
prop.table(table(train_set$Label))

#creating 10 folds cross validation model
cv_folds<-createMultiFolds(train_set$Label, k = 10,times = 1) #creating 10 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 1, index = cv_folds)

#training decision tree model
start_time<-Sys.time()
trained_model_01<-train(Label~., data = train_set, method="rpart", trControl=train_control,
                        tuneLength=7) #Training a decision tree model
total_time<-Sys.time()-start_time
total_time


trained_model_01
#prediction with decision tree model
predictions<-predict(trained_model_01, newdata = test_set[,-1], type = "raw")
confusionMatrix(test_set$Label, predictions, mode = "everything")

#training Support Vector Machine model 
library(e1071)
start_time<-Sys.time()
svm.fit<-svm(Label ~ ., data = train_set, kernel = "linear", cost = 10, scale = FALSE)
total_time<-Sys.time()-start_time
total_time

#prediction with Support vector machine model
predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Label, mode = "everything")

#removing the unused objects and clearing RAM
rm(svm.fit)
rm(trained_model_01)
rm(token.tf)


#IRLBA transformation
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

#merging the label column with IRLBA transformed data
svd.data<-data.frame(Labels=token.tf_idf.df$Label, projected.data$v)
dim(svd.data)

#creating train and test data
test_set<-svd.data[-index,]
train_set<-svd.data[index,]

cv_folds<-createMultiFolds(train_set$Labels, k = 10,times = 1) #creating 30 folds in total 
train_control<-trainControl(method = "repeatedcv", number = 10, repeats = 1, index = cv_folds)



#decision tree model for IRLBA transformed data
start_time<-Sys.time()
trained_model.svd<-train(Labels ~ ., data = train_set, method="rpart",
                         trControl=train_control, tuneLength=7)
total_time<-Sys.time()-start_time
total_time

predictions<-predict(trained_model.svd, test_set[,-1])
confusionMatrix(predictions, test_set$Labels, mode = "everything")

#support vector machine for IRLBA transformed data
library(e1071)
start_time<-Sys.time()
svm.fit<-svm(Labels ~ ., data = train_set)
total_time<-Sys.time()-start_time
total_time

predictions<-predict(svm.fit, test_set[,-1])
confusionMatrix(predictions, test_set$Labels, mode = "everything")


#Deep Learning modeling on SMS Phishing dataset
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

raw_spam_data$Label<-ifelse(raw_spam_data$Label %in% c("smishing","Smishing","Spam","spam") ,"spam","ham")

raw_spam_data$Label<-as.factor(raw_spam_data$Label) #turning Label columns in to factor of two categories
levels(raw_spam_data$Label)

Labels<-raw_spam_data$Label
Labels01<-ifelse(Labels=="ham",1,0)
Labels[1:10]
Labels01[1:10]
save(Labels01, file = "SMS Phishing Labels.RData")

Labels01.encoded<-to_categorical(Labels01)

#BERT model training

start_time<-Sys.time()

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

total_time<-Sys.time()-start_time
total_time

dim(features_train)
saveRDS(object = features_train, file = "feature trained SMS Phishing data.rds")

#creating the deep learning keras API
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




