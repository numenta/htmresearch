# clear work space
rm(list = ls())
suppressMessages(library(TSPred))
setwd('/Users/ycui/nta/nupic.research/sequence_prediction')
dataSet = "SantaFe_A"

if (dataSet == "SantaFe_A"){
  # load Santa Fe Competition Dataset A (Laser Data)
  fitData <- read.table("./data/SantaFe_A.dat")
  testData <- read.table("./data/SantaFe_Acont.dat")
  # fitData <- fitData$V1
  testData <- testData$V1[1:100]
}
if(dataSet == "SantaFeD_"){
  # load Santa Fe Competition Dataset D (Computer generated)
  fitData <- read.table("./data/SantaFe_D1.dat")
  testData <- read.table("./data/SantaFe_D2.dat")
  testData <- fitData$V1[1001:1100]
  fitData <- fitData$V1[1:1000]
}


# Fit ARIMA model and generate prediction
pred <- arimapred(fitData,n.ahead=100)
pred <- as.vector(pred)
ts.cont <- ts(testData,start=1)
yrange = range(range(pred), range(ts.cont))

MSE = mean((pred - ts.cont)^2)

cat("MSE: ", MSE,'\n')
cat("ARIMA predictions saved in 'result/santaFeLaser_ARIMA.csv'",'\n')

# save prediction as a csv file
fileName <- paste0('prediction/', dataSet, '_ARIMA_pred_cont.csv')

df <- data.frame(1:length(pred), pred)
df <- rbind(c('',''), df)
df <- rbind(c('int','float'), df)
df <- rbind(c('step','data'), df)

write.table(df, file=fileName ,sep=',',col.names =FALSE, row.names = FALSE)
write(pred, fileName ,sep=',')

# save prediction as a figure
# pdf('result/santaFeLaser_ARIMA.pdf')
plot(pred, xlim=c(0, length(pred)), ylim=yrange,col='blue',xlab='Time', ylab='Data', type="l")
par(new=T)
plot(ts.cont, xlim=c(0, length(pred)), ylim=yrange,col='red',xlab='Time', ylab='Data')
legend(60, mean(yrange), c("Data","Prediction"),
       lty=c(1,1),lwd=c(2.5,2.5),col=c('blue','red'))
# dev.off()
