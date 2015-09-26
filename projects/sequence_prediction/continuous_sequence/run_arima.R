rm(list = ls())

library(forecast)
library(TSPred)

# set working directory here
setwd('/Users/ycui/nta/nupic.research/projects/sequence_prediction/continuous_sequence')

# available data: "sine", "rec-center-hourly"
dataSet <- 'nyc_taxi'
# dataSet <- 'rec-center-hourly'
# dataSet <- "sine"
dataSetPath <- paste0("data/", dataSet, '.csv')

# load data
recDF <- read.csv(dataSetPath, skip=2)
rt = ts(recDF[2])

if(dataSet=="sine"){
  nTrain <- 1800
} else if (dataSet=='rec-center-hourly'){
  nTrain <- 3800
} else if (dataSet=='nyc_taxi'){
  nTrain <- 6000
}
  
nData <- length(rt)
testLength <- nData - nTrain

# testLength <- 1

# Vectors to hold prediction for t+1
arima_output1 = vector(mode="numeric", length=nData)
arima_output5 = vector(mode="numeric", length=nData)

pred2 <- arimapred(rt[seq(1, nTrain)], n.ahead=testLength)
auto.arima(rt[seq(1, nTrain)])

# Brute force ARIMA - recompute model every step
# while making predictions for the next N hours.
for(i in 1:testLength)
{
  # Compute ARIMA on the full dataset up to this point
  trainSet = window(rt, start=i, end=nTrain+i)
  fit_arima <- auto.arima(trainSet)
  
#   fcast_arima <- predict(fit_arima, n.ahead = 5, se.fit = TRUE)
#   mean <- fcast_arima$pred
#   std <- fcast_arima$se
  
  fcast_arima <- forecast(fit_arima, h=5)
  pred <- fcast_arima$mean

  arima_output1[nTrain+i] = pred[1]
  arima_output5[nTrain+i] = pred[5]
  
  cat("step: ",i ,"true : ", rt[nTrain+i], " prediction: ", pred, '\n')
}

# save prediction as a figure
pdf(paste0('prediction/', dataSet, '_ARIMA_pred.pdf'))
plot(1:testLength, rt[seq(nTrain+1,nTrain+testLength)],'l', col='black',lty=2,xlab='Time', ylab='Data')
lines(1:testLength, arima_output1[seq(nTrain+1,nTrain+testLength)],'l',col='red')
lines(1:testLength, arima_output5[seq(nTrain+1,nTrain+testLength)],'l',col='green')



# Write out output files
# save prediction as a csv file
fileName <- paste0('prediction/', dataSet, '_ARIMA_pred.csv')

df <- data.frame(1:length(rt), rt, arima_output1, arima_output5)
df <- rbind(c('','','',''), df)
df <- rbind(c('int','float','float','float'), df)
df <- rbind(c('step','data','prediction-1step','prediction-5step'), df)

write.table(df, file=fileName ,sep=',',col.names =FALSE, row.names = FALSE)
dev.off()