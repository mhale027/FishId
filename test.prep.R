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

xtest <- matrix(ncol = 45000)
xtest <- matrix(resize(load.image(paste(folder, list.files(folder)[1], sep = "/")), 300,150,1,1), ncol = 45000)
count <- 1
alldata <- matrix(ncol = 45000)
for (i in 1:50) {
      print(i)
      for (j in 1:20) {
            count <- count + 1
            print(count)
            img.r <- load.image(paste(folder, list.files(folder)[count], sep = "/"))
            img.mat <- matrix(resize(img.r, 300, 150, 1, 1), ncol = 45000)
            if (j == 1) {
                  alldata <- img.mat
            } else if (i == 50 & j == 20) {
                  next
            } else {
                  alldata <- rbind(alldata, img.mat)
            }
      }
      
      xtest <- rbind(xtest, alldata)
}



