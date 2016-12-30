library(imager)
library(data.table)
library(h2o)
library(xgboost)
library(caret)
library(dplyr)
library(jpeg)
library(foreach)
library(doParallel)
cl<-makeCluster(8)
registerDoParallel(cl)


setwd("~/projects/kaggle/FishId")
folder <- 'input/train'
print(list.files(folder))
tmp <- lapply(list.files(folder), function(x) list.files(paste0(folder, '/', x)))
files <- data.frame(label = rep(list.files(folder), sapply(tmp, length)), image = unlist(tmp))
fish.files <- filter(files, label != "NoF")

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
      marks$head_x[j] <- as.integer(df$head_x[1])
      marks$tail_x[j] <- as.integer(df$tail_x[1])
      marks$head_y[j] <- as.integer(df$head_y[1])
      marks$tail_y[j] <- as.integer(df$tail_y[1])
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
                  head_x = as.numeric(round(marks$head_x * 300 / dims$rows)),
                  tail_x = as.numeric(round(marks$tail_x * 300 / dims$rows)),
                  head_y = as.numeric(round(marks$head_y * 150 / dims$cols)),
                  tail_y = as.numeric(round(marks$tail_y * 150 / dims$cols))
)
save(box, file = "box.300x150.RData")

allmarked <- alltrain[-grep("NoF", files$label),]
allmarked <- cbind(marks, allmarked)






load("allmarked.RData")
inTrain <- createDataPartition(1:nrow(allmarked), p=.7, list = FALSE)
inValid <- (1:nrow(allmarked))[-inTrain]



h2o.init(nthreads = 8)
h2o.removeAll()


train.hex1 <- as.h2o(allmarked[inTrain[1:1200],], "train.hex1")
train.hex2 <- as.h2o(allmarked[inTrain[1201:2320],], "train.hex2")
train.hex3 <- as.h2o(allmarked[inTrain[1001:1500],], "train.hex3")
train.hex4 <- as.h2o(allmarked[inTrain[1501:2000],], "train.hex4")
train.hex5 <- as.h2o(allmarked[inTrain[2001:2320],], "train.hex5")
train.hex <- rbind(train.hex1, train.hex2, train.hex3, train.hex4, train.hex5)
# train.hex <- as.h20(allmarked[inTrain,], "train.hex")
valid.hex <- as.h2o(allmarked[inValid,], "valid.hex")


t1.hex <- h2o.importFile("h2o.frames", "t1.hex", header = TRUE)
t2.hex <- h2o.importFile("h2o.frame.tr2", "t2.hex", header = TRUE)
t3.hex <- h2o.importFile("h2o.frame.tr3", "t3.hex", header = TRUE)
t4.hex <- h2o.importFile("h2o.frame.tr.4", "t4.hex", header = TRUE)
t5.hex <- h2o.importFile("h2o.frame.tr.5", "t5.hex", header = TRUE)

features <- 6:45005

response <- 2

system.time(dnn.1<-h2o.deeplearning(x = features,
                        y =response,
                        training_frame=train.hex,
                        validation_frame=valid.hex,
                        epochs=1, 
                        stopping_rounds=5,
                        overwrite_with_best_model=T,
                        activation="Rectifier",
                        # distribution="huber",
                        hidden=c(120, 120)))















xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = marks$head_x[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = marks$head_x[inValid])
gc()



xg_eval_mae <- function (yhat, dtrain) {
      y = getinfo(dtrain, "label")
      err= mae(y, yhat)
      return (list(metric = "error", value = err))
}


xgb.params.head_x <- list(
      eta = .1,
      objective = "reg:linear",
      max_depth = 20,
      colsample_bytree = .7,
      subsample = .25,
      base_score = 500
      # feval = xg_eval_mae
)


xgb.head_x <- xgb.train(data = xtrain, 
                        watchlist = list(train = xtrain, valid = xvalid),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.head_x,
                        maximize = FALSE
)

gc()

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = as.integer(marks$tail_x[inTrain]))
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = as.integer(marks$tail_x[inValid]))
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

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = marks$head_y[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = marks$head_y[inValid])
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

xtrain <- xgb.DMatrix(data.matrix(allmarked[inTrain,]), label = marks$tail_y[inTrain])
gc()
xvalid <- xgb.DMatrix(data.matrix(allmarked[inValid,]), label = marks$tail_y[inValid])
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

patches <- list()
species <- list.files("input/train")[-grep("NoF", list.files("input/train"))]
count <- 0
for (i in 1:length(species)) {
      print(i)
      for (j in 1:length(list.files(paste0("input/train/", species[i])))) {
            count <- count + 1
            print(count)
            
            temp.j <- load.image(paste0("input/train/", species[i], "/",
                                        list.files(paste0("input/train/", species[i]))[j]))
            temp.j <- imsub(temp.j, x > min(marks[count,c(2,3)]) & x < max(marks[count,c(2,3)]),
                                    y > min(marks[count,c(4,5)]) & y < max(marks[count,c(4,5)]))
            temp.j <- resize(temp.j, 201, 201, 1, 1)
            if (j == 1) {
                  sum.i <- temp.j
            } else {
                  sum.i <- sum.i + temp.j
            }
      patches[[i]] <- sum.i / length(list.files(paste0("input/train/", species[i])))
      }
}
save(patches, file = "patches.201x201.RData")

for (i in 1:7) {
      patch.i <- patches[[i]]
      if (i == 1) {
            mean.i <- patch.i
      } else {
            mean.i <- mean.i + patch.i
      }
}
mean.patch <- mean.i / 7

patch_size <- 75
patch.edge <- (patch_size*2) + 1
patch.i <- imrotate(resize(mean.patch, patch.edge, patch.edge), 0)
img.i <- load.image(paste0("input/train/", files$label[10], "/", files$image[10]))
img.i <- resize(img.i, dim(img.i)[1], dim(img.i)[2], 1, 1)
search_size <- 25
x1 <- search_size + patch_size + 1
x2 <- dims$rows[1] - search_size - patch_size -1
y1 <- search_size + patch_size +1
y2 <- dims$cols[1] - search_size - patch_size -1
params <- expand.grid(x = seq(x1, x2, search_size), y = seq(y1, y2, search_size))
r  <- foreach(j = 1:nrow(params), .combine=rbind) %dopar% {
      x     <- params$x[j]
      y     <- params$y[j]
      p     <- img.i[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size),1,1]
      score <- cor(as.vector(p), as.vector(patch.i))
      score <- ifelse(is.na(score), 0, score)
      data.frame(x, y, score)}
r <- arrange(r, desc(score))
head(r)