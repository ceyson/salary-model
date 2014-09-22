
## Data source: http://archive.ics.uci.edu/ml/datasets/Census+Income

## Download data
url <- "https://raw.githubusercontent.com/ceyson/salary-model/master/raw-data.csv"
f <- file.path(getwd(), "salary-data.csv")
download.file(url, f)

## Read Data
data <- read.csv("salary-data.csv", 
	             header=T,
	             na.strings="NA",
	             colClass=c('numeric', # age
	             	        'factor',  # workclass
	             	        'numeric', # fnlwgt
	             	        'factor',  # education
	             	        'numeric', # educationNum
	             	        'factor',  # maritalStatus
	             	        'factor',  # occupation
	             	        'factor',  # relationship
	             	        'factor',  # race
	             	        'factor',  # sex
	             	        'numeric', # capitalGain
	             	        'numeric', # capitalLoss
	             	        'numeric', # hours
	             	        'factor',  # nativeCountry
	             	        'factor'   # dv
	             	        ))


## Begin data exploration

## Map missing data by provided feature
require(Amelia)
missmap(data, main="Income Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

## Missingness appears sparse and relegated to workclass and occupation (correlated)
## and native country. It is not clear to me that these would be imputable values so
## I will omit these from the data.

## Remove cases with missing values
nrow(data)
data <- na.omit(data)
nrow(data)

require(Amelia)
missmap(data, main="Income Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)


## Bar plots
attach(data)

barplot(table(workclass),
        names.arg = c(names(summary(workclass))),
        main="Work Class", col="red")

barplot(table(education),
        names.arg = c(names(summary(education))),
        main="Education", col="red")

barplot(table(occupation),
        names.arg = c(names(summary(occupation))),
        main="Occupation", col="red")

barplot(table(relationship),
        names.arg = c(names(summary(relationship))),
        main="Relationship", col="red")

barplot(table(race),
        names.arg = c(names(summary(race))),
        main="Race", col="red")

barplot(table(sex),
        names.arg = c(names(summary(sex))),
        main="Sex", col="red")

barplot(table(dv),
        names.arg = c(names(summary(dv))),
        main="Income (dv)", col="red")

barplot(table(maritalStatus),
        names.arg = c(names(summary(maritalStatus))),
        main="Marital Status", col="red")


## Box plots
boxplot(age ~ dv, 
        main="Income by Age",
        xlab="Income", ylab="Age")

## Mosaic plots
require(vcd)

mosaicplot(education ~ dv, 
           main="Income by Education Level", shade=FALSE, 
           color=TRUE, xlab="Education Level", ylab="Income")

mosaicplot(maritalStatus ~ dv, 
           main="Income by Marital Status", shade=FALSE, 
           color=TRUE, xlab="Marital Status", ylab="Income")

mosaicplot(relationship ~ dv, 
           main="Income by Relationship", shade=FALSE, 
           color=TRUE, xlab="Relationship", ylab="Income")

mosaicplot(race ~ dv, 
           main="Income by Race", shade=FALSE, 
           color=TRUE, xlab="Race", ylab="Income")

mosaicplot(sex ~ dv, 
           main="Income by Sex", shade=FALSE, 
           color=TRUE, xlab="Sex", ylab="Income")

## Summaries
summary(age)
summary(hours)

## Create some features

## Is the individual an immigrant?
data$immigrant <- ifelse(nativeCountry == "United-States", 0, 1)
data$immigrant <- as.factor(data$immigrant)

## Factorize hours of employment
data$employment[hours < 40] <- "Part-time"
data$employment[hours == 40] <- "Full-time"
data$employment[hours > 40] <- "Over-time"
data$employment[hours >= 80] <- "Double-time"
data$employment <- as.factor(data$employment)
str(data$employment)

## Education Level
data$eduLevel[educationNum == 1] <- "Preschool"
data$eduLevel[educationNum == 2] <- "Some Primary School"
data$eduLevel[educationNum == 3] <- "Some Primary School"
data$eduLevel[educationNum == 4] <- "Some Primary School"
data$eduLevel[educationNum == 5 ] <- "Primary School Completed"
data$eduLevel[educationNum == 6] <- "Some High School"
data$eduLevel[educationNum == 7] <- "Some High School"
data$eduLevel[educationNum == 8] <- "Some High School"
data$eduLevel[educationNum == 9 ] <- "High School Graduate"
data$eduLevel[educationNum == 10] <- "Some College"
data$eduLevel[educationNum == 11 ] <- "Associate Vocational Degree"
data$eduLevel[educationNum == 12 ] <- "Associate Academic Degree"
data$eduLevel[educationNum == 13 ] <- "Bachelors Degree"
data$eduLevel[educationNum == 14] <- "Masters Degree"
data$eduLevel[educationNum == 15 ] <- "Professional School"
data$eduLevel[educationNum == 16 ] <- "Doctorate"
data$eduLevel <- as.factor(data$eduLevel)
str(data$eduLevel)


## Plots of features
detach(data)
attach(data)

# Employment
barplot(table(employment),
        names.arg = c(names(summary(employment))),
        main="Employment", col="red")

mosaicplot(employment ~ dv, 
           main="Income by Employment", shade=FALSE, 
           color=TRUE, xlab="Employment", ylab="Income")

# Immigrant
barplot(table(immigrant),
        names.arg = c(names(summary(immigrant))),
        main="Immigrant Status", col="red")

mosaicplot(immigrant ~ dv, 
           main="Income by Immigrant Status", shade=FALSE, 
           color=TRUE, xlab="Immigrant Status", ylab="Income")

# Education Level
barplot(table(eduLevel),
        names.arg = c(names(summary(eduLevel))),
        main="Education Level", col="red")

mosaicplot(eduLevel~ dv, 
           main="Income by Education Level", shade=FALSE, 
           color=TRUE, xlab="Education Level", ylab="Income")

## Convert educationNum to factor
educationNum <- as.factor(educationNum)

## Recode dv
data$dv <- as.factor(ifelse(data$dv == "<=50K", "above50", "below50"))


## Pre-process data
require(caret)

## Check for near-zero variance on continuous variables
continuousVars <- data[,c(1,11,12,13)]
names(continuousVars[,nearZeroVar(continuousVars)])

## Split data
trainingRows <- createDataPartition(dv,
	                                p=.20,
	                                list=FALSE)

train <- data[trainingRows,]
test <- data[-trainingRows,]

## Model Formula
model <- dv ~ age + workclass + maritalStatus + occupation + relationship + race + sex + immigrant + employment + eduLevel

## Parallelization
require(doParallel)

## Logistic
cvCtrl <- trainControl(method="repeatedcv",
                       repeats=5,
                       summaryFunction=twoClassSummary,
                       classProbs=TRUE)

log.fit <- train(model,
                 data=train,
                 method='glm', 
                 family=binomial(link="logit"),
                 metric = "ROC",
                 trControl=cvCtrl)

## Report model fit
log.fit 



## Boosting
gbmtrain <- train
gbmtrain$dv <- as.factor(ifelse(gbmtrain$dv == "below50", 0, 1))

grid <- expand.grid(interaction.depth=seq(1,5,b=2),
	                n.trees=seq(1600,2000,by=50),
	                shrinkage=c(0.01))

ctrl <- trainControl(method="repeatedcv", 
	                 repeats=5,
	                 summaryFunction=twoClassSummary,
	                 classProbs=TRUE)

## Run model in parallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)
ptm <- proc.time()

gbmTune <- train(model,
	             data=gbmtrain,
	             distribution="bernoulli",
	             method="gbm",
	             metric="ROC",
	             tuneGrid=grid,
	             verbose=FALSE,
	             trControl=ctrl)

proc.time() - ptm
stopCluster(cl)

## Report model fit
gbmTune

## Plot performance profile
ggplot(gbmTune) + theme(legend.position="top")

## Random Forest

## Run model in parallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)
ptm <- proc.time()

grid <- expand.grid(.mtry=seq(1,11,b=2))

rfTune <- train(model,
	            data=train,
	            method="rf",
	            metric="ROC",
	            ntree=1000,
	            tuneLength=5,
	            tuneGrid=grid,
	            trControl=ctrl)

proc.time() - ptm
stopCluster(cl)

## Plot performance profile
ggplot(rfTune) + theme(legend.position="top")

## Neural Net

## Run model in parallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)
ptm <- proc.time()

nnetGrid <- expand.grid(.size=1:10,
	                    .decay=c(0,.1,1,2))

maxSize <- max(nnetGrid$.size)
numWts <- 1*(maxSize*(12)+maxSize+1)

nnTune <- train(model,
	            data=train,
	            method="nnet",
	            metric="ROC",
	            preProc=c("center", "scale", "spatialSign"),
	            tuneGrid=nnetGrid,
	            trace=FALSE,
	            maxit=2000,
	            MaxNWts=numWts,
	            trControl=ctrl)

proc.time() - ptm
stopCluster(cl)

## Plot performance profile
ggplot(nnTune) + theme(legend.position="top")




























