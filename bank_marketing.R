#install.packages("h2o", type = "source", repos = (c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))

library(readr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(pander)
library(h2o)
library(knitr)

opts_knit$set(root.dir = "../")

bank <- read.csv("https://www.dropbox.com/s/914qo8t5bhak57y/bank-additional-full.csv?dl=1", sep=";") # specify directory or URL as data soruce

bank <- bank[sample(1:nrow(bank)), ]

colnames(bank)
dim(bank)
str(bank)

pander(summary(bank))

pander(head(bank, 3))

sapply(bank, function(x) sum(is.na(x)))

sapply(bank, function(x) sum(x<0, na.rm=TRUE)) 

pander(bank %>% group_by(y) %>% summarize(n = n()) %>% mutate(percentage = n/sum(n)*100))

pander(summary(bank$age))

ggplot(bank, aes(x = age)) + geom_histogram(binwidth = 5, col = "white") + theme_bw()

nrow(bank[bank$age < 18, ])

nrow(bank[bank$age > 90, ])

bank <- bank %>% filter(age >= 18)
bank <- bank %>% filter(age < 90)
dim(bank)

bank$lnage <- log(bank$age)
pander(summary(bank$lnage))

ggplot(bank, aes(x = lnage)) + geom_histogram(binwidth = 0.1, col = "white") + theme_bw()

ggplot(bank) + geom_histogram(aes(x = age), binwidth = 0.1, col = "white") +
  facet_grid(y~., scales = "free") + scale_x_log10() + theme_bw()

ggplot(bank, aes(x = job)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

nrow(bank[bank$job =="unknown", ])

bank <- bank %>% filter(job != "unknown") 

ggplot(bank) + geom_bar(aes(x = job), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = marital)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

nrow(bank[bank$marital =="unknown", ])

bank <- bank %>% filter(marital != "unknown") 

ggplot(bank) + geom_bar(aes(x = marital), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = education)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

nrow(bank[bank$education =="unknown", ])

nrow(bank[bank$education =="illiterate", ])

ggplot(bank) + geom_bar(aes(x = education), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

bank <- bank %>% filter(education != "illiterate") 

nrow(bank[bank$default =="unknown", ])

ggplot(bank, aes(x = default)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = default), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

nrow(bank[bank$default =="yes", ])

bank <- bank %>% filter(default != "yes") 

ggplot(bank, aes(x = housing)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = housing), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

bank <- bank %>% filter(housing != "unknown") 

ggplot(bank, aes(x = loan)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = loan), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

bank <- bank %>% filter(loan != "unknown") 

ggplot(bank, aes(x = contact)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = contact), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = month)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = month), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = day_of_week)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = day_of_week), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = duration)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ln_duration <- log(bank$duration)
ggplot(bank, aes(x = ln_duration)) + geom_histogram(binwidth = 0.1) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_histogram(aes(x = ln_duration), binwidth = 0.1) +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

summary(bank$campaign)

ggplot(bank, aes(x = campaign)) + geom_histogram(binwidth = 1) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

nrow(bank[bank$campaign > 10, ])

bank <- bank %>% filter(campaign < 10) 

ggplot(bank, aes(x = pdays)) + geom_histogram(binwidth = 50) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = previous)) + geom_histogram(binwidth = 1) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_histogram(aes(x = previous), binwidth = 1) +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = poutcome)) + geom_bar() + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_bar(aes(x = poutcome), col = "white") +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = emp.var.rate)) + geom_histogram(binwidth = 1) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = cons.price.idx)) + geom_histogram(binwidth = 1) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank) + geom_histogram(aes(x = cons.price.idx), binwidth = 1) +
  facet_grid(y~., scales = "free")  + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = cons.conf.idx)) + geom_histogram(binwidth = 5) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = euribor3m)) + geom_histogram(binwidth = 0.01) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(bank, aes(x = nr.employed)) + geom_histogram(binwidth = 100) + theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

dim(bank)

bank$age_1 <- as.numeric(bank$age < 30)
bank$age_2 <- as.numeric(bank$age >= 30 & bank$age <= 60)
bank$age_3 <- as.numeric(bank$age > 60)

bank$job_1 <- as.numeric(bank$job == "admin")
bank$job_2 <- as.numeric(bank$job == "blue_collar")
bank$job_3 <- as.numeric(bank$job == "entrepreneur")
bank$job_4 <- as.numeric(bank$job == "housemaid")
bank$job_5 <- as.numeric(bank$job == "management")
bank$job_6 <- as.numeric(bank$job == "retired")
bank$job_7 <- as.numeric(bank$job == "self-employed")
bank$job_8 <- as.numeric(bank$job == "services")
bank$job_9 <- as.numeric(bank$job == "student")
bank$job_10 <- as.numeric(bank$job == "technician")
bank$job_11 <- as.numeric(bank$job == "unemployed")

bank$marital_1 <- as.numeric(bank$marital == "single")
bank$marital_2 <- as.numeric(bank$marital == "married")
bank$marital_3 <- as.numeric(bank$marital == "divorced")

bank$edu_1 <- as.numeric(bank$education == "basic.4y")
bank$edu_2 <- as.numeric(bank$education == "basic.6y")
bank$edu_3 <- as.numeric(bank$education == "basic.9y")
bank$edu_4 <- as.numeric(bank$education == "high.school")
bank$edu_5 <- as.numeric(bank$education == "professional.course")
bank$edu_6 <- as.numeric(bank$education == "university.degree")
bank$edu_7 <- as.numeric(bank$education == "unknown")

bank$default_1 <- as.numeric(bank$default == "no")
bank$default_2 <- as.numeric(bank$default == "unknown")

bank$housing_1 <- as.numeric(bank$housing == "no")
bank$housing_2 <- as.numeric(bank$housing == "yes")

bank$loan_1 <- as.numeric(bank$loan == "no")
bank$loan_2 <- as.numeric(bank$loan == "yes")

bank$con_1 <- as.numeric(bank$contact == "cellular")
bank$con_2 <- as.numeric(bank$contact == "telephone")

bank$mar <- as.numeric(bank$month == "mar")
bank$apr <- as.numeric(bank$month  == "apr")
bank$may <- as.numeric(bank$month  == "may")
bank$jun <- as.numeric(bank$month  == "jun")
bank$jul <- as.numeric(bank$month  == "jul")
bank$aug <- as.numeric(bank$month  == "aug")
bank$sep <- as.numeric(bank$month  == "sep")
bank$oct <- as.numeric(bank$month  == "oct")
bank$nov <- as.numeric(bank$month  == "nov")
bank$dec <- as.numeric(bank$month  == "dec")

bank$mon <- as.numeric(bank$day_of_week == "mon")
bank$tue <- as.numeric(bank$day_of_week  == "tue")
bank$wed <- as.numeric(bank$day_of_week  == "wed")
bank$thu <- as.numeric(bank$day_of_week  == "thu")
bank$fri <- as.numeric(bank$day_of_week  == "fri")

bank$poutcome1  <- as.numeric(bank$poutcome == "failure")
bank$poutcome2  <- as.numeric(bank$poutcome == "nonexistent")
bank$poutcome3  <- as.numeric(bank$poutcome == "success")

bank <- bank %>% rename(empvarrate = 'emp.var.rate', conspriceidx  = 'cons.price.idx', consconfidx  = 'cons.conf.idx', nremployed  = 'nr.employed')

bank$job <- NULL # Drop originals od dummified categorical features 
bank$marital <- NULL
bank$education <- NULL
bank$default <- NULL
bank$housing <- NULL
bank$loan <- NULL
bank$contact <- NULL
bank$month <- NULL
bank$day_of_week <- NULL
bank$poutcome <- NULL


bank$duration <- NULL
bank$pdays <- NULL

bank$empvarrate <- NULL
bank$conspriceidx <- NULL
bank$consconfidx <- NULL
bank$euribor3m <- NULL
bank$nremployed <- NULL

bank <- bank[,c(4,1,2,3,5:55)] #Reorder variables to put target variable to the first place

colnames(bank)

dim(bank)

set.seed(41)

N <- nrow(bank)
idx_train <- sample(1:N,N/2)
idx_valid <- sample(base::setdiff(1:N, idx_train), N/4)
idx_test <- base::setdiff(base::setdiff(1:N, idx_train),idx_valid)
bank_train <- bank[idx_train,]
bank_valid <- bank[idx_valid,]
bank_test  <- bank[idx_test,]

library(h2o)
h2o.init(port = 58000)

b_train <- as.h2o(bank_train)  

b_valid <- as.h2o(bank_valid)

b_test <- as.h2o(bank_test)

system.time({
  rf <- h2o.randomForest(x = 2:ncol(b_train), y = 1, 
                         seed = 41 ,training_frame = b_train, validation_frame=b_valid,
                         mtries = -1, ntrees = 500, max_depth = 20, nbins = 200)
})

rf

h2o.auc(rf) 

h2o.auc(h2o.performance(rf, b_test))

mse <- h2o.mse(rf,train=TRUE,valid=TRUE)
print(mse)

err <- eval(rf,b_train,b_valid) 
print(err)

system.time({
  gbm <- h2o.gbm(x = 2:ncol(b_train), y = 1, 
                 seed = 41, training_frame = b_train, validation_frame = b_valid,
                 max_depth = 15, ntrees = 500, learn_rate = 0.03, nbins = 100,
                 stopping_rounds = 3)
})

gbm

h2o.auc(gbm)

h2o.auc(h2o.performance(gbm, b_test))

gbm_mse <- h2o.mse(gbm,train=TRUE,valid=TRUE)
print(gbm_mse)

gbm_err <- eval(gbm,b_train,b_valid)
print(gbm_err)

system.time({
  gbmx <- h2o.gbm(x = 2:ncol(b_train), y = 1, 
                  seed = 41, training_frame = b_train, validation_frame = b_valid,
                  max_depth = 15, ntrees = 500, learn_rate = 0.03, nbins = 100, nfolds = 5,
                  stopping_rounds = 3)
})

gbmx

h2o.auc(gbmx)

h2o.auc(h2o.performance(gbmx, b_test))

gbmx_mse <- h2o.mse(gbmx,train=TRUE,valid=TRUE)
print(gbmx_mse)

gbmx_err <- eval(gbmx,b_train,b_valid)
print(gbmx_err)

system.time({
  nn <- h2o.deeplearning(x = 2:ncol(b_train), y = 1, 
                         training_frame = b_train, validation_frame = b_valid,
                         activation = "Rectifier", hidden = c(200,200), epochs = 100,
                         stopping_rounds = 3)
})

nn_mse <- h2o.mse(nn,train=TRUE,valid=TRUE)
print(nn_mse)

nn_err <- eval(nn,b_train,b_valid)
print(nn_err)

######################################## SHINY ########################################

jobList <- c('admin',
             'blue-collar',
             'entrepreneur',
             'housemaid',
             'management',
             'retired',
             'self-employed',
             'services',
             'student',
             'technician',
             'unemployed',
             'unknown')

maritalList <- c('divorced','married','single','unknown')

educationList  <- c('basic.4y',
                    'basic.6y',
                    'basic.9y',
                    'high.school',
                    'illiterate',
                    'professional.course',
                    'university.degree',
                    'unknown')

yesnoList <- c('no',
                 'yes',
                 'unknown')

contactList <- c('cellular','telephone')

dayList <- c('Monday',
             'Tuesday',
             'Wednesday',
             'Thursday',
             'Friday')

monthList <- c('January',
               'February',
               'March',
               'April',
               'May',
               'June',
               'July',
               'August',
               'September',
               'October',
               'November',
               'December'
               )

poutList <- c('Failure',
              'Non-existent',
              'Success')

#runApp("marketingApp", display.mode = "showcase")

gbmpath <- h2o.saveModel(gbm, path = getwd())
rfpath <- h2o.saveModel(rf, path = getwd())
nnpath <- h2o.saveModel(nn, path = getwd())

errgbmpath <- h2o.saveModel(gbm_err, path = getwd())
errrfpath <- h2o.saveModel(err, path = getwd())
errnnpath <- h2o.saveModel(nn_err, path = getwd())
