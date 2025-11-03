install.packages("readr")
library(readr)
setwd("C:/Users/Marina/source/repos/R")
getwd()
list.files()
my_data <- read_csv("hwcData.csv", col_names = FALSE)
file.exists("hwcData.csv")
summary(my_data)     #all staistics (well, almost)
sd(my_data$X1)        # стандартное отклонение/standard deviation
var(my_data$X1)  #variance
nrow(my_data)
hist(my_data$X1,            # first column
     main = " Histogram of monthly household water consumption",
     xlab = "thsd cubic meters",
     ylab = "frequency",
     col = "lightblue",
     border = "black",
     breaks = 40)

