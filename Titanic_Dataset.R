#Loading required libraries
library(dplyr)  # Data Manipulation
library(ggplot2)  # Visualization
library(rattle)  # Visualization
library(RColorBrewer)  # Visualization
library(scales) # Visualization
library(Amelia)  # Visualization
library(car) # Prediction: Checking Multicollinearity
library(caTools) # Prediction: Splitting Data
library(caret)  # Prediction: k-Fold Cross Validation
library(ROCR)  # Prediction: ROC Curve
library(rpart)  # Prediction: Decision Tree
library(rpart.plot)  # Prediction: Decision Tree
library(randomForest)  # Prediction: Random Forest

# Clearing the environment variables
rm(list=ls())

# Set random seed
set.seed(123)

# Set working directory
setwd("D:/Kaggle/Datasets/Titanic")

# Reading the Train data
training <-read.csv(file = 'train.csv',stringsAsFactors = FALSE, header = TRUE)
summary(training)

# Survival rates from train dataset where 0 = Not Survived, 1 = Survived
table(training$Survived)

# Survival rates in propotions from train dataset
prop.table(table(training$Survived))

# Comparision of Sex and Survived column data
table(training$Sex,training$Survived)

# Importing the Test dataset
test<-read.csv(file = 'test.csv',stringsAsFactors = FALSE, header = TRUE)

# Creating a new Survived coulmn in test dataset
test$Survived<-rep(NA,nrow(test))

#Creating a flag to identify data from both Train and Test datasets
training$IsTrainSet <- "TRUE"
test$IsTrainSet <- "FALSE"

# Combining training and test data into a single dataframe
Titanic_data<-rbind(training,test)

# Checking the structure of the data
str(Titanic_data)

# Checking the summary of the data
summary(Titanic_data)

#Removing the training and test data
rm(training)
rm(test)

# Checking missing values or empty cells
colSums(is.na(Titanic_data)|Titanic_data == '')

# Missing data map
missmap(Titanic_data, main = "Titanic data - missing map", col = c("Yellow", "Blue"), legend = TRUE)

# Missing Fare Data Imputation - Replace the missing Fare value with the mean value of same Pclass
Titanic_data$Fare[1044]<-mean(Titanic_data$Fare[which(Titanic_data$Pclass ==3 & is.na(Titanic_data$Fare)==FALSE)])

# Missing Embarked Data Imputation
## Extract the rows which contain the missing Embarked values
filter(Titanic_data, is.na(Embarked)==TRUE|Embarked=='') # Both are Pclass = 1

# Frequency of ports of embarkation of passengers with Pclass = 1.
table(filter(Titanic_data, Pclass ==1)$Embarked)

# Validating the missing data from box plot
ggplot(filter(Titanic_data, is.na(Embarked)==FALSE & Embarked!='' & Pclass==1), 
       aes(Embarked, Fare)) +     
  geom_boxplot(aes(colour = Embarked)) +
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', size=2) +
  ggtitle("Fare distribution of first class passengers") +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5))

# From box plot median fare = 80$ for missing Embarked record can be given 'C' or 'S', I am going with 'C'
Titanic_data$Embarked[which(Titanic_data$Embarked=="")]<-"C"

# Feature Engineering
## Passenger Title
head(Titanic_data$Name)

# Fetch the passenger title from passenger name
Titanic_data$Title <- gsub('(.*, )|(\\..*)', '', Titanic_data$Name)

# To verify list of unique titles
unique(Titanic_data[,14])

# To verify the count of each title
table(Titanic_data$Title)

# Assigning the title based on passenger's Age and gender 
subset(Titanic_data, Title == "Capt") # Age is 70 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Col") # Age is >47 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Don") # Age is 40 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Dona") # Age is 39 & Sex = F, so assigning title as Mrs
subset(Titanic_data, Title == "Jonkheer") # Age is 38 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Lady") # Age is 48 & Sex = F, so assigning title as Mrs
subset(Titanic_data, Title == "Major") # Age is >45 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Mlle") # Age is 24 & Sex = F, so assigning title as Miss
subset(Titanic_data, Title == "Mme") # Age is 24 & Sex = F, so assigning title as Miss
subset(Titanic_data, Title == "Ms") # Age is <28 & Sex = F, so assigning title as Miss
subset(Titanic_data, Title == "Rev") # Age is >28 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "Sir") # Age is 49 & Sex = M, so assigning title as Mr
subset(Titanic_data, Title == "the Countess") # Age is 33 & Sex = F, so assigning title as Mrs

# Fetching Title data
Title_name <- Titanic_data[,c(4,5,6,14)]

# Removing the Title_name dataframe
rm(Title_name)

# Remaning the Title
Titanic_data$Title[Titanic_data$Title == 'Capt'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Col'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Don'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Dona'] <- 'Mrs'
Titanic_data$Title[Titanic_data$Title == 'Jonkheer'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Lady'] <- 'Mrs'
Titanic_data$Title[Titanic_data$Title == 'Major'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Mlle'] <- 'Miss'
Titanic_data$Title[Titanic_data$Title == 'Mme'] <- 'Miss'
Titanic_data$Title[Titanic_data$Title == 'Ms'] <- 'Miss'
Titanic_data$Title[Titanic_data$Title == 'Rev'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'Sir'] <- 'Mr'
Titanic_data$Title[Titanic_data$Title == 'the Countess'] <- 'Mrs'

# To verify the count of each title
table(Titanic_data$Title)

# Data imputation for Age column data
Titanic_data$Age[Titanic_data$Title == 'Dr' & is.na(Titanic_data$Age)==TRUE]<- mean(Titanic_data$Age[Titanic_data$Title == 'Dr' & is.na(Titanic_data$Age)==FALSE])
Titanic_data$Age[Titanic_data$Title == 'Master'& is.na(Titanic_data$Age)==TRUE]<- mean(Titanic_data$Age[Titanic_data$Title == 'Master' & is.na(Titanic_data$Age)==FALSE])
Titanic_data$Age[Titanic_data$Title == 'Miss'& is.na(Titanic_data$Age)==TRUE]<- mean(Titanic_data$Age[Titanic_data$Title == 'Miss' & is.na(Titanic_data$Age)==FALSE])
Titanic_data$Age[Titanic_data$Title == 'Mr'& is.na(Titanic_data$Age)==TRUE]<- mean(Titanic_data$Age[Titanic_data$Title == 'Mr' & is.na(Titanic_data$Age)==FALSE])
Titanic_data$Age[Titanic_data$Title == 'Mrs'& is.na(Titanic_data$Age)==TRUE]<- mean(Titanic_data$Age[Titanic_data$Title == 'Mrs' & is.na(Titanic_data$Age)==FALSE])

# Exploratory Data Analysis
## Encoding the categorical attributes as factors
Titanic_data$Embarked <-as.factor(Titanic_data$Embarked)
Titanic_data$Pclass <-as.factor(Titanic_data$Pclass)
Titanic_data$Sex <-as.factor(Titanic_data$Sex)
Titanic_data$Survived <-as.factor(Titanic_data$Survived)
Titanic_data$N_Family<-Titanic_data$SibSp+Titanic_data$Parch+1

# Checking the Multi collinearity in the data
mcVariables <- c('Survived', 'Pclass', 'Sex', 'Age', 'Sibsp', ' Parch', 'Fare')
multiCol.df <- Titanic_data[,(names(Titanic_data) %in% mcVariables)]
str(multiCol.df)
multiCol.df$Survived <- as.numeric(multiCol.df$Survived)
multiCol.df$Pclass <- as.numeric(multiCol.df$Pclass)
multiCol.df$Sex <- as.numeric(multiCol.df$Sex)
cor(multiCol.df)
pairs(multiCol.df)
rm(mcVariables)
rm(multiCol.df)

# Bar graph for given attributes
par(mfrow=c(1,1))
hist(Titanic_data$Age, freq=F, main='Distribution of Age from full data ', col='darkred', ylim=c(0,0.06))

# Spliting the dataset into Train set and Test set based on flag value
train_data <- Titanic_data[Titanic_data$IsTrainSet==TRUE,]
summary(train_data)

test_data <- Titanic_data[Titanic_data$IsTrainSet==FALSE,]
summary(test_data)

# Creating a child column to validate the survival rate
train_data$chlid <- NA
train_data$chlid[train_data$Age < 18] <- "Child"
train_data$chlid[train_data$Age >= 18] <- "Adult"
table(train_data$chlid, train_data$Survived) # Most of the Children have survived

# Removing the child column
train_data <- train_data[-c(16)]

# Splitting the training dataset into Train set and Validation set
set.seed(101)
Split <- sample.split(train_data$Survived, SplitRatio = 0.7)

train_data <- subset(train_data, Split == TRUE)
validation_data <- subset(train_data, Split == FALSE)

# Removing the Split data
rm(Split)

# Exploratory Data Analysis on Survived and Sex
theme_update(plot.title = element_text(hjust = 0.5))
ggplot(train_data, aes(Survived))+geom_bar(aes(fill = Sex), width = .85, colour = "black")+
  scale_fill_brewer()+
  xlab("Survived: 0= No, 1 = Yes")+
  ylab("Genderwise survival count")+
  ggtitle("Count of People Survived")

# Exploratory Data Analysis on count of Survival of passenger's
ggplot(train_data, aes(Survived))+geom_bar()+ggtitle("Count of People Survived")+geom_bar(color = "blue", fill = "white")+geom_text(stat = 'count', aes(label = ..count..),vjust = 8)
ggplot(train_data, aes(Survived))+geom_bar()+ggtitle("Count of People Survived")+scale_fill_manual(values=c("#E69F00", "#56B4E9"))+geom_text(stat = 'count', aes(label = ..count..),vjust = 0)

# Exploratory Data Analysis on Age, Survived and Sex
ggplot(train_data, aes(Age, fill = factor(Survived))) + geom_histogram() + facet_grid(.~Sex)

# Comparision of Pclass of Train data with gender
ggplot(train_data,aes(x=factor(Pclass),fill=factor(Sex)))+ geom_bar(position="stack")#+geom_text(stat = 'count', aes(label = ..count..), hjust = 1, nudge_x = 0.05, nudge_y = 0.05)

# Comparision of Pclass of full data with gender
ggplot(Titanic_data,aes(x=factor(Pclass),fill=factor(Sex)))+ geom_bar()#+geom_text(stat = 'count', aes(label = ..count..),vjust = 0)

# Survival based on Pclass, Age and Sex (#sourced)
ggplot(filter(Titanic_data, is.na(Survived)==FALSE), aes(Pclass, fill=Survived)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), alpha=0.9, position="dodge") +
  scale_fill_brewer(palette = "Dark2", direction = -1) +
  scale_y_continuous(labels=percent, breaks=seq(0,0.6,0.05)) +
  ylab("Percentage") + 
  ggtitle("Survival Rate based on Pclass") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

# Prediction
## Building Model using Logistic Regression
### Performing Feature Engineering to select the attributes which help prediction.
model_logistic1=glm(Survived~Pclass+Sex+Age+Embarked+N_Family+Fare+Title,family=binomial(link="logit" ),data = train_data)
summary(model_logistic1)
Accuracy1 <- mean(train_data$Survived==round(predict(model_logistic1,train_data,type="response")))
Accuracy1

# Building logistic model without title attribute
model_logistic2=glm(Survived~Pclass+Sex+Age+Embarked+N_Family+Fare,family=binomial(link="logit" ),data = train_data)
summary(model_logistic2)
Accuracy2 <- mean(train_data$Survived==round(predict(model_logistic2,train_data,type="response")))
Accuracy2

# Building logistic model with few attributes
model_logistic3=glm(Survived~Pclass+Sex+Age+Title+Fare,family=binomial(link="logit" ),data = train_data)
summary(model_logistic3)
Accuracy3 <- mean(train_data$Survived==round(predict(model_logistic3,train_data,type="response")))
Accuracy3

# Building logistic model with few other set of attributes
model_logistic4=glm(Survived~Pclass+Sex+Age+N_Family,family=binomial(link="logit" ),data = train_data)
summary(model_logistic4)
Accuracy4 <- mean(train_data$Survived==round(predict(model_logistic4,train_data,type="response")))
Accuracy4

# Removing feature engineering unused models
rm(model_logistic1)
rm(model_logistic2)
rm(model_logistic3)

# Predicting the Train data using glm model with all the important attributes i.e., model_logistic4
Tr_predict <- predict(model_logistic4, train_data, type="response")
train_data$Predicted_Survived<-round(Tr_predict)
write.csv(train_data[,c(2,16)], file = 'Titanic_Train_data_predicted.csv', row.names = F)

confusion_matrix <- table(train_data$Predicted_Survived,train_data$Survived)
confusion_matrix

f.conf <- confusionMatrix((confusion_matrix))
f.conf

# Printing the Accuracy using logistic regression algorithm
accuracy.percent <- 100*sum(diag(confusion_matrix))/sum(confusion_matrix)
print(paste("accuracy:",accuracy.percent,"%"))

# Predicting the Validation set data using glm model with all the important attributes i.e., model_logistic4
Validation_predict <- predict(model_logistic4, validation_data, type="response")
validation_data$Predicted_Survived<-round(Validation_predict)
write.csv(validation_data[,c(2,16)], file = 'Titanic_Validation_data_predicted.csv', row.names = F)

confusion_matrix_v <- table(validation_data$Predicted_Survived,validation_data$Survived)
confusion_matrix_v

f.conf_v <- confusionMatrix((confusion_matrix_v))
f.conf_v

# Printing the Accuracy using Validation data set
accuracy.percent_v <- 100*sum(diag(confusion_matrix_v))/sum(confusion_matrix_v)
print(paste("accuracy:",accuracy.percent_v,"%"))

# Verifying ROC using InformationValue library
InformationValue::plotROC(validation_data$Survived, Validation_predict)
InformationValue::AUROC(validation_data$Survived, Validation_predict)

# Predicting the Test set results
Te_predict <- predict(model_logistic4, test_data, type="response")
test_data$Survived<-round(Te_predict)
write.csv(test_data[,1:2], file = 'Titanic_sol.csv', row.names = F)  # Need to submit output file in kaggle to get the Score

# Removing "Predicted_Survived" column from train_data
train_data <- train_data[-c(16)]

# Removing "Predicted_Survived" column from validation_dataset
validation_data <- validation_data[-c(16)]

# Building Model using Decision Tree Algorithm
## Using Train data
DTfit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
               data=train_data,
               method="class")
graphics.off()
plot(DTfit)
text(DTfit)
fancyRpartPlot(DTfit)

# Prediction using Validation dataset
DT_Prediction_v <- predict(DTfit, validation_data, type = "class")
validation_data$Predicted_Survived_DT<-DT_Prediction_v

confusion_matrix_v_DT <- table(validation_data$Predicted_Survived_DT,validation_data$Survived)
confusion_matrix_v_DT

f.conf_v_DT <- confusionMatrix((confusion_matrix_v_DT))
f.conf_v_DT

# Printing the Accuracy using Validation data set for decision tree model
accuracy.percent_DT <- 100*sum(diag(confusion_matrix_v_DT))/sum(confusion_matrix_v_DT)
print(paste("accuracy:",accuracy.percent_DT,"%"))

# Predicting using Test dataset
DT_Prediction1 <- predict(DTfit, test_data, type = "class")
submit1 <- data.frame(PassengerId = test_data$PassengerId, Survived = DT_Prediction1)
write.csv(submit1, file = "DecisionTree.csv", row.names = FALSE) # Need to submit output file in kaggle to get the Score

# Removing "Predicted_Survived_DT" column from validation_dataset
validation_data <- validation_data[-c(16)]

# Building Model using Random Forest Algorithm
set.seed(101)
train_data$Title <- as.factor(train_data$Title)
validation_data$Title <- as.factor(validation_data$Title)
rfFit <- randomForest(as.factor(Survived) ~ Pclass +  Sex + Age + SibSp + Parch + Fare + Embarked + Title + N_Family,
                      data = train_data,
                      mtry = 3)
print(rfFit)

# Prediction using Random Forest model on Train data
rfPredict <- predict(rfFit, train_data)
confusionMatrix(rfPredict, train_data$Survived)

# Validating the important Variables for data
varImpPlot(rfFit, main = "Variable Importance")
varImpPlot(rfFit, sort = T, n.var = 5, main = "Top 5 - Variable Importance")
#varUsed(rfFit) #Variables used in ranfom forest 'rfFit'

# In Importance, Type = 1 is for accuracy, MeanDecreaseAccuracy graph, Measures how worse the model performs without each variable.
imp_accuracy <- importance(rfFit, type = 1)
imp_accuracy

# In Importance, Type = 2 is for Gini index, MeanDecreaseGini graph, Measures how pure the nodes are at the end of the tree without each variable.
imp_nodeImpurity <- importance(rfFit, type = 2)
imp_nodeImpurity 

# Predicting using Validation dataset on Random Forest Algorithm
levels(validation_data$Title)<-levels(train_data$Title) #We observed the levels mismatch in train and validation set data
RF_Prediction_v <- predict(rfFit, validation_data)
validation_data$Predicted_Survived_RF<-RF_Prediction_v

confusion_matrix_v_RF <- table(validation_data$Predicted_Survived_RF,validation_data$Survived)
confusion_matrix_v_RF

f.conf_v_RF <- confusionMatrix((confusion_matrix_v_RF))
f.conf_v_RF

# Printing the Accuracy using Validation data set for Random Forest model
accuracy.percent_RF <- 100*sum(diag(confusion_matrix_v_RF))/sum(confusion_matrix_v_RF)
print(paste("accuracy:",accuracy.percent_RF,"%"))

# Prediction using Random Forest model on Test data
test_data$Title <- as.factor(test_data$Title)
rfPrediction <- predict(rfFit, test_data)
rfSubmit <- data.frame(PassengerId = test_data$PassengerId, Survived = rfPrediction)
write.csv(rfSubmit, file = "Titanic_Prediction using RandomForest.csv", row.names = FALSE) # Need to submit output file in kaggle to get the Score
