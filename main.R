library(imager)
library(data.table)
library(h2o)
library(xgboost)
library(caret)
library(dplyr)
library(jpeg)

setwd("~/projects/kaggle/FishId")
folder <- 'input/test_stg1'
tests <- list.files(folder)



for (i in 1:length(tests)) {
      img.i <- load.image(paste(folder, tests[i], sep = "/"))
      img.r <- resize(img.i, 300, 150, 1, 1)
      save.image(img.r, file = paste0("test.prep/", tests[i]))
}

count <- 0
alldata <- matrix(ncol = 45000)
folder <- "input/test_stg1"
for (i in 1:20) {
      print(i)
      for (j in 1:50) {
            count <- count + 1
            print(count)
            img.r <- load.image(paste(folder, list.files(folder)[count], sep = "/"))
            img.mat <- matrix(resize(img.r, 300, 150, 1, 1), ncol = 45000)
            if (j == 1) {
                  alldata <- img.mat
            } else {
                  alldata <- rbind(alldata, img.mat)
            }
      }
      if (i == 1) {
            alltest <- alldata
      } else {
            alltest <- rbind(alltest, alldata)
      }
}

save(alltest, file = "alltest.300x150.RData")



count <- 0
alldata <- matrix(ncol = 45000)
folder <- "input/train"
for (i in 1:76) {
      print(i)
      for (j in 1:50) {
            count <- count + 1
            print(count)
            img.r <- load.image(paste(folder, files$label[count], files$image[count], sep = "/"))
            img.mat <- matrix(resize(img.r, 300, 150, 1, 1), ncol = 45000)
            if (j == 1) {
                  alldata <- img.mat
            } else if (i == 76 && j >= 28) {
                  next
            } else {
                  alldata <- rbind(alldata, img.mat)
            }
      }
      if (i == 1) {
            alltrain <- alldata
      } else {
            alltrain <- rbind(alltrain, alldata)
      }
}
alltrain <- rbind(alltrain, alldata)
save(alltrain, file = "alltrain.300x150.final.RData")


folder <- 'input/train'
print(list.files(folder))
tmp <- lapply(list.files(folder), function(x) list.files(paste0(folder, '/', x)))
files <- data.frame(label = rep(list.files(folder), sapply(tmp, length)), image = unlist(tmp))
markers <- data.frame(fread("markers.csv"))





isfish <- c(rep(1, length(list.files("alldata"))))
isfish[grep("NoF", list.files("alldata"))] <- 0

classes <- factor(unique(files$label))
labels <- as.numeric(factor(files$label))


markers[,2:5] <- sapply(markers[,2:5], as.numeric)
marks <- data.frame(image = fish.files$image,
                    head_x = rep(0, nrow(fish.files)),
                    tail_x = rep(0, nrow(fish.files)),
                    head_y = rep(0, nrow(fish.files)),
                    tail_y = rep(0, nrow(fish.files)))


for (j in 1:nrow(fish.files)) {
      df <- filter(markers, image == fish.files$image[j])
      marks$head_x[j] <- min(df[1,c(2,4)]) - 50
      marks$tail_x[j] <- max(df[1,c(2,4)]) + 50
      marks$head_y[j] <- min(df[1,c(3,5)]) - 50
      marks$tail_y[j] <- max(df[1,c(3,5)]) + 50
}
save(marks, file = "marks.RData")


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

box <- data.frame(image = marks$image, 
                  head_x = marks$head_x * 300 / dims$rows,
                  tail_x = marks$tail_x * 300 / dims$rows,
                  head_y = marks$head_y * 150 / dims$cols,
                  tail_y = marks$tail_y * 150 / dims$cols
)
save(box, file = "box.300x150.RData")

allmarked <- alltrain[-grep("NoF", files$label),]

inTrain <- createDataPartition(1:nrow(box), p=.9, list = FALSE)
inValid <- (1:nrow(box))[-inTrain]

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$head_x[inTrain])
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$head_x[inValid])

xgb.params.head_x <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.head_x <- xgb.train(data = xtrain, 
                        watchlist = list(train = xtrain, valid = xvalid),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.head_x
)

gc()

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$tail_x[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$tail_x[inValid])
gc()
xgb.params.tail_x <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.tail_x <- xgb.train(data = xtrain, 
                        watchlist = list(train = xtrain, valid = xvalid),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.tail_x
)
gc()

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$head_y[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$head_y[inValid])
gc()
xgb.params.head_y <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.head_y <- xgb.train(data = xtrain, 
                        watchlist = list(train = xtrain, valid = xvalid),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.head_y
)

gc()

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = box$tail_y[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = box$tail_y[inValid])
gc()
xgb.params.tail_y <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 15,
      colsample_bytree = .3,
      subsample = .25
)


xgb.tail_y <- xgb.train(data = xtrain, 
                        watchlist = list(train = xtrain, valid = xvalid),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.tail_y
)
gc()
save(xgb.head_x, xgb.tail_x, xgb.head_y, xgb.tail_y, file = "box.set.RData")
gc()
xtrain <- xgb.DMatrix(data.matrix(allmarked))

pred.train <- data.frame(image = marks$image,
                         head_x = predict(xgb.head_x, xtrain), 
                         head_y = predict(xgb.head_y, xtrain),
                         tail_x = predict(xgb.tail_x, xtrain),
                         tail_y = predict(xgb.tail_y, xtrain)
)
gc()


inTrain <- createDataPartition(1:nrow(files), p=.9, list = FALSE)
inValid <- (1:nrow(files))[-inTrain]



xtrain <- xgb.DMatrix(data.matrix(alltrain[inTrain,]), label = isfish[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(alltrain[inValid,]), label = isfish[inValid])
gc()
xgb.params.hasfish <- list(
      eta = .01,
      objective = "multi:softmax",
      num_class = 2,
      max_depth = 20,
      colsample_bytree = .5,
      subsample = .5
)


xgb.widefish <- xgb.train(data = xtrain, 
                          watchlist = list(train = xtrain, valid = xvalid),
                          nrounds = 500,
                          early_stopping_rounds = 30,
                          params = xgb.params.hasfish
)







xtest <- xgb.DMatrix(data.matrix(alltest))

pred.test.isfish <- predict(xgb.widefish, xtest)

pred.test <- data.frame(image = list.files("input/test_stg1"),
                        head_x = predict(xgb.head_x, xtest),
                        head_y = predict(xgb.head_y, xtest),
                        tail_x = predict(xgb.tail_x, xtest),
                        tail_y = predict(xgb.tail_y, xtest)
)
write.table(pred.train, 'train.markers.csv', quote = FALSE, sep = ',', row.names = FALSE)
write.table(pred.test, 'test.markers.csv', quote = FALSE, sep = ',', row.names = FALSE)


test.image <- function(id) {
      plot(resize(load.image(paste0("input/test_stg1/", list.files("input/test_stg1")[id])), 300, 150, 1 ,1))
      points(pred.test[id,c(2,4)], pred.test[id,c(3,5)], col = "red", lwd = 2)
      
}

train.image <- function(id) {
      plot(resize(load.image(paste0("input/train/", files$label[id], "/", files$image[id])), 300, 150, 1 ,1))
      points(pred.train[id,c(2,4)], pred.train[id,c(3,5)], col = "red", lwd = 2)
      
}

means <- list()
species <- list.files("input/train")
for (i in 1:length(species)) {
      print(i)
      for (j in 1:length(grep(species[i], list.files("alldata")))) {
            print(j)
            if (i == 1) {
                  temp.i <- load.image(list.files("alldata")[i])
            } else {
                  
            }
      }
}




