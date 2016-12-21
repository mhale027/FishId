library(imager)
library(data.table)
library(h2o)
library(xgboost)
library(caret)
library(dplyr)
library(jpeg)
library(imageData)

setwd("~/projects/kaggle/Fish")

folder <- 'input/train'
print(list.files(folder))
tmp <- lapply(list.files(folder), function(x) list.files(paste0(folder, '/', x)))
files <- 
      data.frame(label = rep(list.files(folder), sapply(tmp, length)),
                 image = unlist(tmp))



dm <- matrix(rep(0, 5e6), ncol = 5e6)
img1 <- load.image(paste(folder, files[1,1], files[1,2], sep = "/"))
imgr <- resize(img1, round(dim(img1)[1]/4), round(dim(img1)[2]/4), 1, 1)
imgr.mat <- matrix(imgr, nrow = 1)
mat1 <- matrix(c(rep(0, 100), imgr.mat, rep(0, (5e6 - length(imgr.mat)-100))), nrow = 1)

for (i in 2:nrow(files)) {
      
      img1 <- load.image(paste(folder, files[i,1], files[i,2], sep = "/"))
      imgr <- resize(img1, round(dim(img1)[1]/4), round(dim(img1)[2]/4), 1, 1)
      imgr.mat <- matrix(imgr, nrow = 1)
      mat1 <- rbind(mat1, c(rep(0, 100), imgr.mat, rep(0, (5e6 - length(imgr.mat)-100))))
      
}
      
#       
#          image   head_x   head_y    tail_x   tail_y
# 1: img_00003.jpg 307.5738 380.7634  600.6306 191.4169


dims <- data.frame(rows = 0, cols = 0, depth = 0, channels = 0)
for (i in 1:nrow(files)) {
      dims[i,] <- dim(load.image(paste(folder, files[i,1], files[i,2], sep = "/")))
}
      
save(dims, file = "dims.RData")
      
      
      
      
      
load("markers.csv")
markers <- data.frame(markers)
files <- data.frame(files)
for (i in 1:nrow(files)) {
      
      file <- files$image[i]
      
      img.i <- load.image(paste(folder, files[i,1], files[i,2], sep = "/"))
      
      marks <- filter(markers, image == file)
      imgs <- list()
      for (j in 1:nrow(marks)) {
            if (nrow(marks) == 0) {
                  next
            } else {
                  xmin <- min(marks[j,c(2,4)]) - 50
                  xmax <- max(marks[j,c(2,4)]) + 50
                  ymin <- min(marks[j,c(3,5)]) - 50
                  ymax <- max(marks[j,c(3,5)]) + 50
            
                  img.i.j <- imsub(img.i, x < xmax & x > xmin, y < ymax & y > ymin)
                  img.i.j <- resize(img.i.j, round(dim(img.i.j)[1]/4), round(dim(img.i.j)[2]/4), 1, 1)
                  imgs[[j]] <- matrix(img.i.j, nrow = dim(img.i.j)[1])
            }
      }
            
      for (k in 1:length(imgs)) {
            if (length(imgs) == 0 ) {
                  writeJPEG(matrix(resize(img.i, dim(img.i)[1], dim(img.i)[2],1,1)/255, nrow = dim(img.i)[1]),
                            raw(), 
                            bg = "black",
                            target = paste0("fish.proc/", files[i,"label"],"_",i, ".empty.jpg"), quality = 1)
            } else {
                  writeJPEG(imgs[[k]]/255, 
                            raw(), 
                            bg = "black",
                            target = paste0("fish.proc/", files[i,"label"],"_",i, ".", k, ".jpg"), quality = 1)
            }
      }

}




















