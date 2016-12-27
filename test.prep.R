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


