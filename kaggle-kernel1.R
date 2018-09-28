#=============================================
# Bank Marketing
#=============================================

# Load dataset from working directory
df <- read.csv("bank.csv", sep = ';')

# Exploratory analysis
dim(df)
str(df)
summary(df)
df <- df[order(df$y),] # ordering by 'y'
levels(df$y) <- c(0,1) # no-0, yes-1

# Missing values
sum(is.na(df))


# Numeric and factor split 
dfn <- df[,sapply(df, function(x) is.numeric(x))]
str(dfn)
summary(dfn)
dfc <- df[,sapply(df, is.factor)]
str(dfc)
summary(dfc)

# visualisation
for (i in (1:ncol(dfc))){
  plot(dfc[,i], col = rainbow(20), main = names(dfc)[i])
}

for (i in (1:ncol(dfn))){
  hist(dfn[,i], col = rainbow(20), main = names(dfn)[i])
  boxplot(dfn[,i], col = rainbow(20), main = names(dfn)[i])
}



# Data normalisation: Scaling the numeric variables using max-min method
df$pdays[df$pdays==-1] <- 0
for (i in c(1:ncol(df))){
  if (sapply(df[i], is.numeric))
    df[i] <- (max(df[i])-df[i])/max(df[i])
}

dfn <- df[,sapply(df, is.numeric)]
str(dfn)
summary(dfn)
dfn <- cbind(dfn, df$y)


# Feature construction: F-score to calculate interaction effects
fs <- matrix(0,ncol(dfn)-1,ncol(dfn)-1) # Empty square matrix
v <- 0
for (i in 1:ncol(fs)){
  for (j in 1:i){
    v <-0
    v <- dfn[,i]*dfn[,j] # product of 2 columns
    meanpos <- mean(v[c(0:4000)])
    meanneg <- mean(v[c(4001:4521)])
    meantot <- mean(v[c(0:4521)])
    varpos <- var(v[c(0:4000)])
    varneg <- var(v[c(4001:4521)])
    fs[i,j] <- (((meanpos-meantot)^2) + ((meanneg-meantot)^2))/(varpos+varneg) # F-score
  }
}

# selection of important features after examining the F-scores
a <- cbind.data.frame(df$duration*df$previous, df$duration^2 , df$duration*df$balance, 
                      df$duration*df$campaign, df$duration*df$pdays)
names(a) <- c('dur_prev','dur2','dur_bal','dur_camp','dur_pdays')
df1 <- cbind.data.frame(df, a)



# Feature Selection using Random Forest 
library(randomForest)
rffit <- randomForest(data = df1, df1$y~., ntree = 500, mtry = 12, importance = T)
importance(rffit)
varImpPlot(rffit) # The attribute and gini plot give the relative importance of the 
# features towards the model.

# Feature Selection: Boruta
library(Boruta)
battrib <- Boruta(data = df, df$y~.)
battrib # Important, tentative and rejected features shown



# Class Balancing
plot(df$y, col = rainbow(20), main = "Success rate") # There is an imbalance.
non <- df1[df1$y==0,] # non-responders, y=0
res <- df1[df1$y==1,] # responders, y=1
indexnon <- sample(1:nrow(non), 2*nrow(res), replace = F) # non-responders taken twice 
# as much as responders
train <- rbind(res, non[indexnon,])
str(train)



# Splitting our dataset into test and train
train <- train[,-c(9,11,14,15,16,18,22)] # Test dataset doesn't accomodate these features. 
# Can be observed after EDA.
s <- sample(nrow(train), 0.6*nrow(train), replace = F)
strain <- train[s,]
stest <- train[-s,]



# SVM model 1, radial kernel, on 60% train
library(e1071)
wts <- 100/table(strain$y) # Consider weights
svmfit1 <- tune(svm, data = strain, y~., class.weights = wts, probability = T, # Tuning model for best fit
                ranges = list(gamma = 2^(-8:0), cost = 10^(-2:4)), scale = F)

radfit1 <- svmfit1$best.model # Best fit(gamma and cost) model used
summary(radfit1)                
pred <- predict(radfit1, newdata = stest[,-12]) # Prediction on split test
table(predict = pred, truth = stest$y) # Cross-tab 
sum(pred==stest$y)/length(stest$y) # Accuracy



#===================================================================
# Test
#===================================================================

# Loading test data set, EDA and all the previous analysis performed
# Load dataset from working directory
dftest <- read.csv("test.csv")

# Exploratory analysis
dim(dftest)
str(dftest)
summary(dftest)

# Missing values
sum(is.na(dftest)) 

# Numeric and factor split 
dftestn <- dftest[,sapply(dftest, function(x) is.numeric(x))]
str(dftestn)
summary(dftestn)
dftestc <- dftest[,sapply(dftest, is.factor)]
str(dftestc)
summary(dftestc)

# visualisation
for (i in (1:ncol(dftestc))){
  plot(dftestc[,i], col = rainbow(20), main = names(dftestc)[i])
}

for (i in (1:ncol(dftestn))){
  hist(dftestn[,i], col = rainbow(20), main = names(dftestn)[i])
  boxplot(dftestn[,i], col = rainbow(20), main = names(dftestn)[i])
}




# Data normalisation: Scaling the numeric variables using max-min method
dftest$pdays[dftest$pdays==-1] <- 0
for (i in c(1:ncol(dftest))){
  if (sapply(dftest[i], is.numeric))
    dftest[i] <- (max(dftest[i])-dftest[i])/max(dftest[i])
}

a2 <- cbind.data.frame(dftest$duration*dftest$previous, dftest$duration^2 , dftest$duration*dftest$balance, 
                       dftest$duration*dftest$campaign, dftest$duration*dftest$pdays)
names(a2) <- c('dur_prev','dur2','dur_bal','dur_camp','dur_pdays')
dftest <- cbind.data.frame(dftest, a2)
dftest <- (subset(dftest, select = -Id))





#====================================================================================
# SVM Model2: radial kernel, trained on complete train, applied on given test dataset
#====================================================================================

wts <- 100/table(df$y) # Consider weights
svmfit2 <- tune(svm, data = df, y~., class.weights = wts, probability = T, # Tuning model for best fit
                ranges = list(gamma = 2^(-8:0), cost = 10^(-2:4)), scale = F)

radfit2 <- svmfit2$best.model # Best fit(gamma and cost) model used
summary(radfit2)                
pred_test <- predict(radfit2, newdata = dftest) # Prediction on given test




# Submission as a csv file
pred_test <- as.data.frame(pred_test)
write.csv(pred_test,"tested01.csv")
dftest <- dftest[,-c(16,11,9,15,14,21,17)]