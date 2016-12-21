library(shiny)
library(imager)
library(data.table)

setwd("~/projects/kaggle/Fish")

folder <- 'input/train'
print(list.files(folder))
tmp <- lapply(list.files(folder), function(x) list.files(paste0(folder, '/', x)))
files <- 
      data.table(label = rep(list.files(folder), sapply(tmp, length)),
                 image = unlist(tmp))

files <- files[label != 'test_stg1']

#Choose one:
# 1: new file
# markers <- data.table(image= character(), head_x = integer(), head_y = integer(), tail_x = integer(), tail_y = integer())
# 2: load file
markers <- fread('markers.csv')

# use this to save (after closing app)
#write.table(markers, 'markers.csv', quote = FALSE, sep = ',', row.names = FALSE)


ui <- bootstrapPage(
      plotOutput('image',
                 click = 'click',
                 width = '900px',
                 height = '600px'
      ),
      actionButton('save', 'save'),
      actionButton('clear', 'clear'),
      actionButton('back', 'back'),
      numericInput('row', 'go to image', 1),
      actionButton('go', 'go')
)

server <- function(input, output) {
      obj <- reactiveValues(
            heads_x = c(),
            heads_y = c(),
            tails_x = c(),
            tails_y = c(),
            i = 1,
            img = NULL
      )

      observeEvent(input$go, {obj$i <- input$row})

      observe({
            img <- files[obj$i, image]
            obj$img = load.image(paste(folder, files[obj$i, label], img, sep = '/'))
            my <- markers[image == img]
            obj$heads_x <- my[, head_x]
            obj$heads_y <- my[, head_y]
            obj$tails_x <- my[, tail_x]
            obj$tails_y <- my[, tail_y]
      })

      observeEvent(input$click, {
            if (length(obj$heads_x) == length(obj$tails_y)) {
                  obj$heads_x <- c(obj$heads_x, input$click$x)
                  obj$heads_y <- c(obj$heads_y, input$click$y)
            } else {
                  obj$tails_x <- c(obj$tails_x, input$click$x)
                  obj$tails_y <- c(obj$tails_y, input$click$y)
            }
      })

      observeEvent(input$save, {
            if (length(obj$heads_x) > 0) {
                  img = files[obj$i, image]
                  markers <<- rbind(markers[image != img],
                                    data.table(image = img,
                                               head_x = obj$heads_x,
                                               head_y = obj$heads_y,
                                               tail_x = obj$tails_x,
                                               tail_y = obj$tails_y),
                                    fill = TRUE)
            }
            obj$i <- obj$i + 1
      })

      observeEvent(input$back, {
            obj$i <- obj$i - 1
      })

      observeEvent(input$clear, {
            obj$heads_x <- obj$heads_y <- obj$tails_x <- obj$tails_y <- c()
      })

      output$image <- renderPlot({
            lab <- files[obj$i, label]
            img <- files[obj$i, image]
            plot(obj$img)
            title(paste(lab, img, 'id = ',obj$i))
            points(obj$heads_x, obj$heads_y, col = 'blue', lwd = 3, pch = 16)
            points(obj$tails_x, obj$tails_y, col = 'red', lwd = 3, pch = 16)
      })
}

#run this!
shinyApp(ui = ui, server = server)