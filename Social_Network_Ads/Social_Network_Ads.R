# Logistic Regression on Social Network Ads
library(ggplot2)
library(caTools)
library(caret)
library(boot)
library(dplyr)
library(pROC)
library(ROCR)
library(InformationValue)
library(rpart)

set.seed(123)
#Set Working Directory
setwd("D:/Kaggle/Datasets/Social_Network_Ads")

# Importing the dataset
dataset = read.csv(file = 'Social_Network_Ads.csv')
summary(dataset) #No missing values in given dataset
str(dataset)
# Performing EDA on complete dataset
#Binning or creating categories for Age feature into 3 categories
dataset$Bins_Age <- cut(dataset$Age,breaks = 3, labels = c("Young","Adult","Senior"))

#Creating datafames for 3 age categories
Young_DF <- subset(dataset, Bins_Age=="Young")
Adult_DF <- subset(dataset, Bins_Age=="Adult")
Senior_DF <- subset(dataset, Bins_Age=="Senior")

# Bar graph for given attributes
theme_update(plot.title = element_text(hjust = 0.5))
ggplot(data=Young_DF,aes(x= Age, y=EstimatedSalary))+geom_bar(stat = "identity")+xlab("Young User")+ggtitle("Estimated Salary for Young User")
ggplot(Adult_DF,aes(x= Age, y=EstimatedSalary))+geom_bar(stat = "identity")+xlab("Adult User")+ggtitle("Estimated Salary for Adult User")
ggplot(Senior_DF,aes(x= Age, y=EstimatedSalary))+geom_bar(stat = "identity")+xlab("Senior User")+ggtitle("Estimated Salary for Senior User")

ggplot(dataset,aes(Bins_Age))+geom_bar()+xlab("Age groups")+ggtitle("Count of Users according to Age Groups")
ggplot(dataset,aes(Gender))+geom_bar()+ggtitle("Count of Users according to Gender")
ggplot(dataset,aes(Age))+geom_bar()+ggtitle("Count of Users for all Age group users")
ggplot(dataset,aes(EstimatedSalary))+geom_bar()+ggtitle("Count of users Estimated Salary")
ggplot(dataset,aes(Purchased))+geom_bar()+ggtitle("Classification of Product Not Purchased vs Purchased")

# Scatter plots to check the correlation between age and Estimated salary
pairs(dataset[,3:4]) # This plot is used only for continous data - Age, Salary, Scatter plot shows no correlation between Age and Salary
str(dataset)
table(dataset$Purchased) #0->257, 1->143 Imbalanced class
prop.table(table(dataset$Purchased)) #0-> 64,25%, 1-> 35.75%

# Selecting necessary attributes for model
dataset = dataset[2:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Logistic Regression to the Training set
mylogit1 = glm(formula = Purchased ~ ., family = binomial(link = "logit"), data = training_set)
summary(mylogit1)

# Predicting the Test set results
pred1 = predict(mylogit1, newdata = test_set[-4], type = 'response')

#checking accuracy
y_pred1 = ifelse(pred1 > 0.5, 1, 0)
error <- mean(y_pred1 !=test_set$Purchased)
print(paste('Full model accuracy is ', 1-error))

#another method to validate the accuracy
y_act <- test_set$Purchased
mean(y_pred1 == y_act)

#caret::confusionMatrix(y_pred1, y_act, positive="1")

# Making the Confusion Matrix
confusion_matrix = table(test_set[, 4], y_pred1 > 0.5)
confusion_matrix

#Using InformationValue library
InformationValue::plotROC(y_act, pred1)
InformationValue::AUROC(y_act, pred1)

#Validating the model using K-Fold Cross Validation technique on Train dataset

#Define train control for K-fold Cross Validation
train_control <- trainControl(method = "cv", number = 5, savePredictions = TRUE) #k=5 fold closs validation

#Fitting K-Fold Cross validation model on Full model
cv.mylogit1 = train(Purchased~., data = training_set, method = "glm",trControl = train_control)
cv.mylogit1 # Validate the accuracy

#predictions <- predict(cv.mylogit,pred) # Issue in this step as the cv.mylogit is a list

########### Building Logistic regression model without Gender attribute ################

# Fitting Logistic Regression to the Training set
mylogit2 = glm(formula = Purchased ~ Age+EstimatedSalary, family = binomial(link = "logit"), data = training_set)
summary(mylogit2)

# Predicting the Test set results
pred2 = predict(mylogit2, newdata = test_set[2:3], type = 'response')

#checking accuracy
y_pred2 = ifelse(pred2 > 0.5, 1, 0)
error <- mean(y_pred2 !=test_set$Purchased)
print(paste('Accuracy without Gender attribute is ', 1-error))

#another method to validate the accuracy
y_act <- test_set$Purchased
mean(y_pred2 == y_act)

#caret::confusionMatrix(y_pred2, y_act, positive="1")

# Making the Confusion Matrix
confusion_matrix = table(test_set[, 4], y_pred2 > 0.5)
confusion_matrix

#Using InformationValue library
InformationValue::plotROC(y_act, pred2)
InformationValue::AUROC(y_act, pred2)

#Validating the model using K-Fold Cross Validation technique on Train dataset without Gender variable

#Fit K-Fold Cross validation model
cv.mylogit2 = train(Purchased~Age+EstimatedSalary, data = training_set, method = "glm",trControl = train_control)
cv.mylogit2 # Validate the accuracy

predictions <- predict(cv.mylogit2,pred) # Issue in this step as the cv.mylogit is a list

#############Downsampling technique for Imbalanced class####

set.seed(123)
down_train <- downSample(x=training_set, y=training_set$Purchased)
table(down_train$Purchased)
down_train<-down_train[-5]

# Fitting Logistic Regression to the Training set
mylogit3 = glm(formula = Purchased ~ Age+EstimatedSalary, family = binomial(link = "logit"), 
               data = down_train)
summary(mylogit3)

# Predicting the Test set results
pred3 = predict(mylogit3, newdata = test_set[-4], type = 'response')

#checking accuracy
y_pred3 = ifelse(pred3 > 0.5, 1, 0)
error <- mean(y_pred3 !=test_set$Purchased)
print(paste('Model accuracy after downsampling data and without Gender variable is ', 1-error))

#another method to validate the accuracy
y_act <- test_set$Purchased
mean(y_pred3 == y_act)

#caret::confusionMatrix(y_pred, y_act, positive="1")

# Making the Confusion Matrix
confusion_matrix = table(test_set[, 4], y_pred3 > 0.5)
confusion_matrix

#Using InformationValue library
InformationValue::plotROC(y_act, pred3)
InformationValue::AUROC(y_act, pred3)

## Using downSample data, validating K-Fold Cross validation model 
#Fit K-Fold Cross validation model
cv.mylogit3 = train(Purchased~Age+EstimatedSalary, data = down_train, method = "glm",trControl = train_control)
cv.mylogit3 # Validate the accuracy

#############Upsampling technique for Imbalanced class####

set.seed(123)
up_train <- upSample(x=training_set, y=training_set$Purchased)
table(up_train$Purchased)
up_train<-up_train[-5]

# Fitting Logistic Regression to the Training set
mylogit4 = glm(formula = Purchased ~ Age+EstimatedSalary, family = binomial(link = "logit"), 
               data = up_train)
summary(mylogit4)

# Predicting the Test set results
pred4 = predict(mylogit4, newdata = test_set[-4], type = 'response')

#checking accuracy
y_pred = ifelse(pred4 > 0.5, 1, 0)
error <- mean(y_pred !=test_set$Purchased)
print(paste('Model accuracy after Upsampling data and without Gender variable is ', 1-error))

#another method to validate the accuracy
y_act <- test_set$Purchased
mean(y_pred == y_act)

#caret::confusionMatrix(y_pred, y_act, positive="1")

# Making the Confusion Matrix
confusion_matrix = table(test_set[, 4], y_pred > 0.5)
confusion_matrix

#Using InformationValue library
InformationValue::plotROC(y_act, pred4)
InformationValue::AUROC(y_act, pred4)

## Using upSample data, validating K-Fold Cross validation model 
#Fit K-Fold Cross validation model
cv.mylogit4 = train(Purchased~Age+EstimatedSalary, data = up_train, method = "glm",trControl = train_control)
cv.mylogit4 # Validate the accuracy

###########Building Decision Trees classification model##
library(rpart.plot)
dtm1 <- rpart(Purchased~Gender+Age+EstimatedSalary, data = training_set, method = "class")
test_set1 <- test_set[2:4]
rpart.plot(dtm1, extra = 106)

#making the prediction
Prediction1 <- predict(dtm1, test_set, type = "class")
#test_set$Gender <- factor(test_set$Gender)

Table_matrix1 <- table(test_set$Purchased, Prediction1)
Table_matrix1

Accuracy_test1 <- sum(diag(Table_matrix1))/sum(Table_matrix1)
print(paste("Accuracy of test using Decision trees is: ",Accuracy_test1))

#Tune of Hyper-parameters
Accuracy_Tuning1 <- function(dtm1){
  Prediction1 <- predict(dtm1, test_set, type = "class")
  Table_matrix1 <- table(test_set$Purchased, Prediction1)
  Accuracy_test1 <- sum(diag(Table_matrix1))/sum(Table_matrix1)
  Accuracy_test1
}

Trn_control1 <- rpart.control(minsplit = 4, 
                              minbucket = round(5/3),
                              maxdepth = 3,
                              cp = 0)

tune_dtm1 <- rpart(Purchased~Gender+Age+EstimatedSalary, data = training_set, method = "class", control = Trn_control1)
Accuracy_Tuning1(tune_dtm1) # Accuracy Increased from 0.86 in DT model to 0.89 in tuning Hyper parameters

