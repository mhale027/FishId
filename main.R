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
markers <- read.csv("markers.csv")



for (i in 1:nrow(markers)) {
  fold <- files[files$image == as.character(markers$image[i]),]$label
  img <- load.image(paste0("input/train/", fold, "/",markers$image[i]))
  x.center <- (markers[i,2] + markers[i,4])/2
  y.center <- (markers[i,3] + markers[i,5])/2
  dx <- markers[i,2] - markers[i,4]
  dy <- markers[i,3] - markers[i,5]
  d <- max(abs(dx), abs(dy))/2
  
  img1 <- imsub(img, x > (x.center - d - 5) & x < (x.center + d + 5))
  img1 <- imsub(img1, y > (y.center - d - 5) & y < (y.center + d + 5))
  img1 <- resize(img1, dim(img1)[1], dim(img1)[2], 1, 1)
  # mg1 <- imrotate(img1, -atan(dy/dx)*180/pi)
  # if (dx > 0) { img1 <- mirror(img1, "x") }
  # img2 <- imsub(img1, x > (dim(img1)[1]/10) & x < (dim(img1)[1]*9/10))
  # img2 <- imsub(img2, y > (dim(img2)[2]/4) & y < (dim(img2)[2]*3/4))
  save.image(img1, paste0("outputs/pos/", i, ".jpg"))
}

for (i in 1:length(list.files("input/train/NoF"))) {
  img1 <- load.image(paste0("input/train/NoF/", list.files("input/train/NoF")[i]))
  img2 <- resize(img1, dim(img1)[1], dim(img1)[2], 1, 1)
  save.image(img2, paste0("outputs/neg/", i, ".jpg"))
}

fish <- list.files("input/train")[-c(grep("NoF", list.files("input/train")))]
for (j in 1:length(fish)) {
  path.i <- paste0("input/train/", fish[j], "/")
  count <- 1
  for (i in 1:length(list.files(path.i))) {
    m <- markers[markers$image == as.character(files[files$label == fish[j],]$image[i]),]
    for (k in 1:nrow(m)) {
      if (nrow(m) == 0) {next} else {
        img <- load.image(paste0(path.i, list.files(path.i)[i]))
        img <- resize(img, dim(img)[1], dim(img)[2], 1, 1)
        
        x.center <- (m[k,2] + m[k,4])/2
        y.center <- (m[k,3] + m[k,5])/2
        dx <- m[k,2] - m[k,4]
        dy <- m[k,3] - m[k,5]
        d <- max(abs(dx), abs(dy))/2
      
        img <- imsub(img, x > (x.center - d - 5) & x < (x.center + d + 5))
        img <- imsub(img, y > (y.center - d - 5) & y < (y.center + d + 5))
      
      
      
        save.image(img, paste0("outputs/pos", fish[j], "/", count, ".jpg"))
        count <- count + 1
      }
    }
  }
}




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



grads <- data.frame(matrix(rep(0, 45000),ncol = 45000))
names(grads) <- paste("pixel_", 1:45000)



for (i in 1:nrow(files)) {
      print(i)
      im <- load.image(paste("input/train", files$label[i], files$image[i], sep = "/"))
      im <- get_gradient(resize(isoblur(im, 4), 150, 75, 1, 1), "xy")
      im <- sqrt((im[[1]]^2) + (im[[2]]^2))
      save.image(im, paste0("gradients.150x75/", files$label[i], "_", files$image[i]))
      grads[i,] <- matrix(im, nrow = 1)
}
save(grads, file = "grads.150x75.RData")




for (i in 1:nrow(files)) {
      print(i)
      im <- load.image(paste("input/train", files$label[i], files$image[i], sep = "/"))
      im <- get_gradient(isoblur(grayscale(im), 5), "xy")
      im <- sqrt((im[[1]]^2) + (im[[2]]^2))
      save.image(im, paste0("gradients.wide.blur/", files$label[i], "_", files$image[i]))
}
save(grads.wide, file = "grad.wide.blur.RData")














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
for (i in 1:nrow(marks)) {
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
save(box, file = "box300x150.RData")

allmarked <- alltrain[-grep("NoF", files$label),]
allmarked <- cbind(marks, allmarked)






load("allmarked.RData")
inTrain <- createDataPartition(1:nrow(allmarked), p=.7, list = FALSE)
inValid <- (1:nrow(allmarked))[-inTrain]



h2o.init(nthreads = 8)
h2o.removeAll()


# train.hex1 <- as.h2o(allmarked[inTrain[1:1200],], "train.hex1")
# train.hex2 <- as.h2o(allmarked[inTrain[1201:2320],], "train.hex2")
# train.hex3 <- as.h2o(allmarked[inTrain[1001:1500],], "train.hex3")
# train.hex4 <- as.h2o(allmarked[inTrain[1501:2000],], "train.hex4")
# train.hex5 <- as.h2o(allmarked[inTrain[2001:2320],], "train.hex5")
# train.hex <- rbind(train.hex1, train.hex2, train.hex3, train.hex4, train.hex5)
# train.hex <- as.h2o(allmarked[inTrain,], "train.hex")
# valid.hex <- as.h2o(allmarked[inValid,], "valid.hex")

# t1.hex <- h2o.importFile("h2o.frames", "t1.hex", header = TRUE)
# t2.hex <- h2o.importFile("h2o.frame.tr2", "t2.hex", header = TRUE)
# t3.hex <- h2o.importFile("h2o.frame.tr3", "t3.hex", header = TRUE)
# t4.hex <- h2o.importFile("h2o.frame.tr.4", "t4.hex", header = TRUE)
# t5.hex <- h2o.importFile("h2o.frame.tr.5", "t5.hex", header = TRUE)

train.path <- paste0(getwd(), "/", "train.hex")
valid.path <- paste0(getwd(), "/", "valid.hex")
train.hex <- h2o.importFile(path = train.path, destination_frame = "train.hex", header = TRUE)
valid.hex <- h2o.importFile(path = valid.path, destination_frame = "valid.hex", header = TRUE)

features <- 6:45005

response <- 2

system.time(dnn<-h2o.deeplearning(x = features,
                        y =response,
                        training_frame=train.hex,
                        validation_frame=valid.hex,
                        epochs=1, 
                        stopping_rounds=5,
                        overwrite_with_best_model=T,
                        activation="Rectifier",
                        # distribution="huber",
                        hidden=c(200,200)))

h2o.shutdown()





















xgb.grid <- expand.grid(
      nrounds = 2, 
      eta = 0.05, 
      max_depth = seq(10,18, 1), 
      gamma = c(0,1,2,5), 
      colsample_bytree = c(0.2, 0.3), 
      min_child_weight = c(1)
)

xgb.train.control <- trainControl(
      method = "cv",
      number = 2,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep <- train(x=data.matrix(allmarked[1:100,-c(1:5)]),
                  y=allmarked[1:100,2],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree"
)
















xgb.params <- list(
      eta = .03,
      objective = "reg:linear",
      max_depth = 13,
      colsample_bytree = .2,
      subsample = .4
      # feval = xg_eval_mae
)


train1 <- 1:1656
train2 <- 1657:3312


xtrain1 <- xgb.DMatrix(data.matrix(allmarked[train1,-c(1:5)]), label = box$head_x[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked[train2,-c(1:5)]), label = box$head_x[train2])
gc()
xgb.params <- list(
      eta = .03,
      objective = "reg:linear",
      max_depth = 13,
      colsample_bytree = .2,
      subsample = .4,
      base_score = 122
)
xgb.head_x.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        maximize = FALSE,
                        print_every_n = 20
)
gc()
xgb.head_x.2 <- xgb.train(data = xtrain2, 
                          watchlist = list(train = xtrain2, valid = xtrain1),
                          nrounds = 500,
                          early_stopping_rounds = 10,
                          params = xgb.params,
                          maximize = FALSE,
                          print_every_n = 20
)
gc()









xtrain1 <- xgb.DMatrix(data.matrix(allmarked[train1,-c(1:5)]), label = as.integer(box$tail_x[train1]))
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked[train2,-c(1:5)]), label = as.integer(box$tail_x[train2]))
gc()
xgb.params <- list(
      eta = .03,
      objective = "reg:linear",
      max_depth = 13,
      colsample_bytree = .2,
      subsample = .4,
      base_score = 156
)
xgb.tail_x.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 20
)
gc()
xgb.tail_x.2 <- xgb.train(data = xtrain2, 
                        watchlist = list(train = xtrain2, valid = xtrain1),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 20
)
gc()





xtrain1 <- xgb.DMatrix(data.matrix(allmarked[train1,-c(1:5)]), label = box$head_y[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked[train2,-c(1:5)]), label = box$head_y[train2])
gc()
xgb.params <- list(
      eta = .03,
      objective = "reg:linear",
      max_depth = 13,
      colsample_bytree = .2,
      subsample = .4,
      base_score = 75
)
xgb.head_y.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 20
)
gc()
xgb.head_y.2 <- xgb.train(data = xtrain2, 
                        watchlist = list(train = xtrain2, valid = xtrain1),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 20
)
gc()




xtrain1 <- xgb.DMatrix(data.matrix(allmarked[train1,-c(1:5)]), label = box$tail_y[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked[train2,-c(1:5)]), label = box$tail_y[train2])
gc()
xgb.params <- list(
      eta = .03,
      objective = "reg:linear",
      max_depth = 13,
      colsample_bytree = .2,
      subsample = .4,
      base_score = 78
)
xgb.tail_y.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 10
)
gc()
xgb.tail_y.2 <- xgb.train(data = xtrain2, 
                        watchlist = list(train = xtrain2, valid = xtrain1),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params,
                        print_every_n = 10
)
gc()


# save(xgb.head_x, xgb.tail_x, xgb.head_y, xgb.tail_y, file = "xgb.marks.set.RData")
# gc()
# xtrain <- xgb.DMatrix(data.matrix(allmarked))

pred.train <- data.frame(image = allmarked$image,
                         head_x = c(predict(xgb.head_x.2, xtrain1), predict(xgb.head_x.1, xtrain2)),
                         head_y = c(predict(xgb.head_y.2, xtrain1), predict(xgb.head_x.1, xtrain2)),
                         tail_x = c(predict(xgb.tail_x.2, xtrain1), predict(xgb.head_x.1, xtrain2)),
                         tail_y = c(predict(xgb.tail_y.2, xtrain1), predict(xgb.head_x.1, xtrain2))
)
gc()









allmarked.l2 <- cbind(pred.train, allmarked[,-c(1:5)])



######################################################################
########################## level 2 ###################################
######################################################################

xgb.grid.l2 <- expand.grid(
      nrounds = 2, 
      eta = 0.05, 
      max_depth = seq(4,20, 1), 
      gamma = c(0,1,2), 
      colsample_bytree = c(0.2, 0.3, 1), 
      min_child_weight = c(1,10,25,100)
)

xgb.train.control.l2 <- trainControl(
      method = "cv",
      number = 2,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep.l2 <- train(x=data.matrix(allmarked.l2[1:100,-c(1)]),
                  y=allmarked.l2[1:100,2],
                  trControl = xgb.train.control.l2,
                  tuneGrid = xgb.grid.l2,
                  method = "xgbTree"
)



















xtrain1 <- xgb.DMatrix(data.matrix(allmarked.l2[train1,-c(1)]), label = box$head_x[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked.l2[train2,-c(1)]), label = box$head_x[train2])
gc()


xgb.params.l2 <- list(
      eta = .05,
      objective = "reg:linear",
      max_depth = 16,
      colsample_bytree = .2,
      gamma = 2,
      subsample = .4,
      base_score = 122
)
xgb.head_x.l2.1 <- xgb.train(data = xtrain1, 
                           watchlist = list(train = xtrain1, valid = xtrain2),
                           nrounds = 500,
                           early_stopping_rounds = 10,
                           params = xgb.params.l2,
                           maximize = FALSE,
                           print_every_n = 20
)
gc()
xgb.head_x.l2.2 <- xgb.train(data = xtrain2, 
                               watchlist = list(train = xtrain2, valid = xtrain1),
                               nrounds = 500,
                               early_stopping_rounds = 10,
                               params = xgb.params.l2,
                               maximize = FALSE,
                               print_every_n = 20
)
gc()






xtrain1 <- xgb.DMatrix(data.matrix(allmarked.l2[train1,-c(1)]), label = box$tail_x[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked.l2[train2,-c(1)]), label = box$tail_x[train2])
gc()
xgb.params.l2 <- list(
      eta = .05,
      objective = "reg:linear",
      max_depth = 16,
      colsample_bytree = .2,
      gamma = 2,
      subsample = .4,
      base_score = 156
)
xgb.tail_x.l2.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.l2,
                        print_every_n = 20
)
gc()
xgb.tail_x.l2.2 <- xgb.train(data = xtrain2, 
                           watchlist = list(train = xtrain2, valid = xtrain1),
                           nrounds = 500,
                           early_stopping_rounds = 10,
                           params = xgb.params.l2,
                           print_every_n = 20
)
gc()




xtrain1 <- xgb.DMatrix(data.matrix(allmarked.l2[train1,-c(1)]), label = box$head_y[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked.l2[train2,-c(1)]), label = box$head_y[train2])
gc()
xgb.params.l2 <- list(
      eta = .05,
      objective = "reg:linear",
      max_depth = 16,
      colsample_bytree = .2,
      gamma = 2,
      subsample = .4,
      base_score = 75
)
xgb.head_y.l2.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.l2,
                        print_every_n = 20
)
gc()
xgb.head_y.l2.2 <- xgb.train(data = xtrain2, 
                           watchlist = list(train = xtrain2, valid = xtrain1),
                           nrounds = 500,
                           early_stopping_rounds = 10,
                           params = xgb.params.l2,
                           print_every_n = 20
)
gc()




xtrain1 <- xgb.DMatrix(data.matrix(allmarked.l2[train1,-c(1)]), label = box$tail_y[train1])
gc()
xtrain2 <- xgb.DMatrix(data.matrix(allmarked.l2[train2,-c(1)]), label = box$tail_y[train2])
gc()
xgb.params.l2 <- list(
      eta = .05,
      objective = "reg:linear",
      max_depth = 16,
      colsample_bytree = .2,
      gamma = 2,
      subsample = .4,
      base_score = 78
)
xgb.tail_y.l2.1 <- xgb.train(data = xtrain1, 
                        watchlist = list(train = xtrain1, valid = xtrain2),
                        nrounds = 500,
                        early_stopping_rounds = 10,
                        params = xgb.params.l2,
                        print_every_n = 20
)
gc()

xgb.tail_y.l2.2 <- xgb.train(data = xtrain2, 
                           watchlist = list(train = xtrain2, valid = xtrain1),
                           nrounds = 500,
                           early_stopping_rounds = 10,
                           params = xgb.params.l2,
                           print_every_n = 20
)
gc()


save(xgb.head_x.l2, xgb.tail_x.l2, xgb.head_y.l2, xgb.tail_y.l2, file = "xgb.marks.l2.set.RData")
gc()


xtrain <- xgb.DMatrix(data.matrix(allmarked))

pred.train.l2 <- data.frame(image = allmarked.l2$image,
                         head_x = c(predict(xgb.head_x.l2.2, xtrain1), predict(xgb.head_x.l2.1, xtrain2)),
                         head_y = c(predict(xgb.head_y.l2.2, xtrain1), predict(xgb.head_x.l2.1, xtrain2)),
                         tail_x = c(predict(xgb.tail_x.l2.2, xtrain1), predict(xgb.head_x.l2.1, xtrain2)),
                         tail_y = c(predict(xgb.tail_y.l2.2, xtrain1), predict(xgb.head_x.l2.1, xtrain2))
)
gc()



pred.train.wide <- data.frame(image = allmarked.l2$image, 
                              head_x = as.numeric(round(pred.train$head_x * dims$rows / 300)),
                              tail_x = as.numeric(round(pred.train$tail_x * dims$rows / 300)),
                              head_y = as.numeric(round(pred.train$head_y * dims$cols / 150)),
                              tail_y = as.numeric(round(pred.train$tail_y * dims$cols / 150))
)

pred.train.l2.wide <- data.frame(image = allmarked.l2$image, 
                              head_x = as.numeric(round(pred.train.l2$head_x * dims$rows / 300)),
                              tail_x = as.numeric(round(pred.train.l2$tail_x * dims$rows / 300)),
                              head_y = as.numeric(round(pred.train.l2$head_y * dims$cols / 150)),
                              tail_y = as.numeric(round(pred.train.l2$tail_y * dims$cols / 150))
)






########################################################################
########################################################################
############################## crop images #############################
########################################################################

means <- data.frame(image = allmarked.l2$image)
means$markers_x <- (marks$head_x + marks$tail_x) / 2
means$markers_y <- (marks$head_y + marks$tail_y) / 2
means$pred_x <- (pred.train.wide$head_x + pred.train.wide$tail_x) / 2
means$pred_y <- (pred.train.wide$head_y + pred.train.wide$tail_y) / 2
means$pred.l2_x <- (pred.train.l2.wide$head_x + pred.train.l2.wide$tail_x) / 2
means$pred.l2_y <- (pred.train.l2.wide$head_y + pred.train.l2.wide$tail_y) / 2
means <- mutate(means, px = (pred_x + pred.l2_x)/2)
means <- mutate(means, py = (pred_y + pred.l2_y)/2)

allcut <- data.frame(matrix(rep(0, 45000), nrow = 1))
names(allcut) <- paste0("pixel_", 1:45000)
for (i in 1:nrow(allmarked.l2)) {
      print(i)
      img.i <- load.image(paste("input/train", fish.files$label[i], fish.files$image[i], sep = "/"))
      img.i <- imsub(img.i, x > means$px[i] - 300 & x < means$px[i] + 300)
      img.i <- imsub(img.i, y > means$py[i] - 150 & y < means$py[i] + 150)
      save.image(img.i, file = paste0("allcut1/", fish.files$label[i], "_", fish.files$image[i]))
      img.i <- resize(img.i, 300, 150, 1, 1)
      allcut[i,] <- data.frame(matrix(img.i, nrow = 1))
}
save(allcut, "allcut1.RData")

crop.markers <- box   
deltas <- data.frame(image = allmarked.l2$image, dx = rep(0, nrow(allmarked.l2)), dy = rep(0, nrow(allmarked.l2)))
for (i in 1:nrow(allmarked.l2)) {
      
      crop.markers$head_x[i] <- (marks$head_x[i] - means$px[i] + 300) / 2
      crop.markers$tail_x[i] <- (marks$tail_x[i] - means$px[i] + 300) / 2
      
      crop.markers$head_y[i] <- (marks$head_y[i] - means$py[i] + 150) / 2
      crop.markers$tail_y[i] <- (marks$tail_y[i] - means$py[i] + 150) / 2
      
      
}
      
cropped.image <- function(id) {
      plot(resize(load.image(paste0("allcut1/", fish.files$label[id], "_", fish.files$image[id])), 300, 150, 1 ,1))
      points(crop.markers[id,c(2,4)], crop.markers[id,c(3,5)], col = "red", lwd = 2)
      
}      





deg <- function(rad) 180*rad/pi




for (i in 1:nrow(markers)) {
      img <- load.image(paste0("train/", markers$image[i]))
      x.center <- mean(markers[i,2], markers[i,4])
      y.center <- mean(markers[i,3], markers[i,5])
      dx <- markers[i,2] - markers[i,4]
      dy <- markers[i,3] - markers[i,5]
      d <- max(abs(dx), abs(dy))/2
      
      img1 <- imsub(img, x > min(markers[i,2], markers[i,4]), x < max(markers[i,2], markers[i,4]))
      img1 <- imsub(img1, y > min(markers[i,3], markers[i,5]), y < max(markers[i,3], markers[i,5]))
      img1 <- imrotate(img1, -atan(dy/dx)*180/pi)
      save.image(img1, paste0("cv/pos/", i, ".jpg"))
}
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

for (i in 1:nrow(markers)) {
      img <- load.image(paste0("train/", markers$image[i]))
      x.center <- (markers[i,2] + markers[i,4])/2
      y.center <- (markers[i,3] + markers[i,5])/2
      dx <- markers[i,2] - markers[i,4]
      dy <- markers[i,3] - markers[i,5]
      d <- max(abs(dx), abs(dy))/2

      img1 <- imsub(img, x > (x.center - d - 20) & x < (x.center + d + 20))
      img1 <- imsub(img1, y > (y.center - d - 20) & y < (y.center + d + 20))
      img1 <- resize(img1, 30 + dim(img1)[1]/10, 30 + dim(img1)[2]/10, 1, 1)
      # img1 <- imrotate(img1, -atan(dy/dx)*180/pi)
      # if (dx > 0) { img1 <- mirror(img1, "x") }
      # img2 <- imsub(img1, x > (dim(img1)[1]/2 - d) & x < (dim(img1)[1]/2 + d))
      # img2 <- imsub(img2, y > (dim(img1)[2]/2 - d) & y < (dim(img1)[2]/2 + d))
      save.image(img1, paste0("cv/cuts/", i, ".jpg"))
}
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
for (i in 1:nrow(markers)) {
      img <- resize(load.image(paste0("cv/pos40x40/", i, ".jpg")), 60, 50, 1, 1)
      img1 <- imsub(img, x > 5 & x < 55)
      img2 <- imsub(img1, y > 10 & y < 40)
      save.image(img2, paste0("cv/pos50x30/", i, ".jpg"))
}




















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
      plot(resize(load.image(paste0("input/train/", fish.files$label[id], "/", fish.files$image[id])),300,150,1,1))
      points(box$head_x[id], box$head_y[id], col = "red", lwd = 2)
      points(box$tail_x[id], box$tail_y[id], col = "blue", lwd = 2)
      points(pred.train$head_x[id], pred.train$head_y[id], col = "red", pch = 19, lwd = 5)
      points(pred.train$tail_x[id], pred.train$tail_y[id], col = "blue", pch = 19, lwd = 5)
      points(pred.train.l2$head_x[id], pred.train.l2$head_y[id], col = "red", pch = 23, lwd = 10)
      points(pred.train.l2$tail_x[id], pred.train.l2$tail_y[id], col = "blue", pch = 23, lwd = 10)
}




test.wide <- function(id) {
      plot(load.image(paste0("input/test_stg1/", list.files("input/test_stg1")[id])))
      points(pred.test[id,c(2,4)], pred.test[id,c(3,5)], col = "red", lwd = 2)
      
}

train.wide <- function(id) {
      plot(load.image(paste0("input/train/", fish.files$label[id], "/", fish.files$image[id])))
      points(allmarked$head_x[id], allmarked$head_y[id], col = "red", lwd = 2)
      points(allmarked$tail_x[id], allmarked$tail_y[id], col = "blue", lwd = 2)
      points(pred.train.wide$head_x[id], pred.train.wide$head_y[id], col = "red", pch = 19, lwd = 5)
      points(pred.train.wide$tail_x[id], pred.train.wide$tail_y[id], col = "blue", pch = 19, lwd = 5)
      points(pred.train.l2.wide$head_x[id], pred.train.l2.wide$head_y[id], col = "red", pch = 23, lwd = 10)
      points(pred.train.l2.wide$tail_x[id], pred.train.l2.wide$tail_y[id], col = "blue", pch = 23, lwd = 10)
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