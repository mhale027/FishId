library(imager)
library(data.table)
library(h2o)
library(xgboost)
library(caret)
library(dplyr)
library(jpeg)

setwd("~/projects/kaggle/FishId")

folder <- 'input/train'
print(list.files(folder))
tmp <- lapply(list.files(folder), function(x) list.files(paste0(folder, '/', x)))
files <- data.frame(label = rep(list.files(folder), sapply(tmp, length)), image = unlist(tmp))
markers <- data.frame(fread("markers.csv"))
marks <- data.frame(image = markers$image,
                    xmin = rep(0, nrow(markers)),
                    xmax = rep(0, nrow(markers)),
                    ymin = rep(0, nrow(markers)),
                    ymax = rep(0, nrow(markers)))

for (j in 1:nrow(markers)) {
      marks$xmin[j] <- min(markers[j,c(2,4)]) - 20
      marks$xmax[j] <- max(markers[j,c(2,4)]) + 20
      marks$ymin[j] <- min(markers[j,c(3,5)]) - 20
      marks$ymax[j] <- max(markers[j,c(3,5)]) + 20
}


cuts <- c(FALSE)
for (i in 1:nrow(marks)) {
      if (i == 1) {
            next
      }else {
            if (marks$image[i] == marks$image[i-1]) {
                  cuts[i] <- TRUE
            } else {
                  cuts[i] <- FALSE
            }
      }
}


solomarks <- marks[!cuts,]
save(solomarks, "solomarks.RData")

fish.files <- filter(files, label != "NoF")
folder <- "input/train"
dims <- data.frame(label = 0, image = 0 , rows = 0, cols = 0, depth = 0, channels = 0)
for (i in 1:nrow(solomarks)) {
      print(i)
      dims[i,] <- c(as.character(fish.files$label[i]), as.character(fish.files$image[i]), dim(load.image(paste(folder, 
                                                                                   fish.files$label[i],
                                                                                   fish.files$image[i],
                                                                                   sep = "/"))))
}
dims$rows <- as.numeric(dims$rows)
dims$cols <- as.numeric(dims$cols)
save(dims, file = "dims.RData")

box <- data.frame(image = solomarks$image, 
                  xmin = solomarks$xmin * 300 / dims$rows,
                  xmax = solomarks$xmax * 300 / dims$rows,
                  ymin = solomarks$ymin * 150 / dims$cols,
                  ymax = solomarks$ymax * 150 / dims$cols
)
save(box, file = "box.300x150.RData")

allmarked <- all[-grep("NoF", files$image),]

inTrain <- createDataPartition(1:nrow(box), p=.7, list = FALSE)
inValid <- (1:nrow(box))[-inTrain]

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$xmin[inTrain])
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$xmin[inValid])

xgb.params.xmin <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.xmin <- xgb.train(data = xtrain, 
                          watchlist = list(train = xtrain, valid = xvalid),
                          nrounds = 500,
                          early_stopping_rounds = 30,
                          params = xgb.params.xmin
)


xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$xmax[inTrain])
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$xmax[inValid])

xgb.params.xmax <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.xmax <- xgb.train(data = xtrain, 
                      watchlist = list(train = xtrain, valid = xvalid),
                      nrounds = 500,
                      early_stopping_rounds = 30,
                      params = xgb.params.xmax
)


xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$ymin[inTrain])
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$ymin[inValid])

xgb.params.ymin <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.ymin <- xgb.train(data = xtrain, 
                      watchlist = list(train = xtrain, valid = xvalid),
                      nrounds = 500,
                      early_stopping_rounds = 30,
                      params = xgb.params.ymin
)



xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$ymax[inTrain])
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$ymax[inValid])

xgb.params.ymax <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.ymax <- xgb.train(data = xtrain, 
                      watchlist = list(train = xtrain, valid = xvalid),
                      nrounds = 500,
                      early_stopping_rounds = 30,
                      params = xgb.params.ymax
)

save(xgb.xmin, xgb.xmax, xgb.ymin, xgb.ymax, file = "box.set.RData")

xtrain <- xgb.DMatrix(data.matrix(allmarked))
xtest <- xgb.DMatrix(data.matrix(alltest))

pred.train <- data.frame(image = solomarks$image,
                         head_x = predict(xgb.xmin, xtrain), 
                         head_y = predict(xgb.ymin, xtrain),
                         tail_x = predict(xgb.xmax, xtrain),
                         tail_y = predict(xgb.ymax, xtrain)
)

pred.test <- data.frame(image = list.files("input/test_stg1"),
                        head_x = predict(xgb.xmin, xtest),
                        head_y = predict(xgb.ymin, xtest),
                        tail_x = predict(xgb.xmax, xtest),
                        tail_y = predict(xgb.ymax, xtest)
)
write.table(pred.train, 'train.markers.csv', quote = FALSE, sep = ',', row.names = FALSE)
write.table(pred.test, 'test.markers.csv', quote = FALSE, sep = ',', row.names = FALSE)
