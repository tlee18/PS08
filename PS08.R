# Tim Lee
# Date: 10-31-17
# Problem Set 8
# Collaborated with: Pei Gong, Jonathan Che

library(tidyverse)
library(caret)

# Package for easy timing in R
library(tictoc)


# Demo of timer function --------------------------------------------------
# Run the next 5 lines at once
tic()
Sys.sleep(3)
timer_info <- toc()
runtime <- timer_info$toc - timer_info$tic
runtime



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/PS08/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)



# Time knn here -----------------------------------------------------------

n_values <- seq(from = 0.01, to = 0.2, by = .01)
k_values <- seq(from = 1, to = 10000, by = 1000)




# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point
runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)
runtime_dataframe



time <- function(runtime_dataframe=runtime_dataframe, train){
  for(i in 1:nrow(runtime_dataframe)){
    sampleSize <- runtime_dataframe$n[i] * nrow(train)
    trainSample <- slice(train, 1:sampleSize)
    tic()
    model_knn <- caret::knn3(model_formula, data=trainSample,
                             k = runtime_dataframe$k[i])
    clock <- toc()
    runtime_dataframe$runtime[i] <- clock$toc - clock$tic
    }
  return(runtime_dataframe)
}


runtime_dataframe <- time(runtime_dataframe = runtime_dataframe, train = train)

# Plot 1 shows that the runtime increases as n increases. As we move horizontally, we see the runtime increase steadily. 
# However, runtime as a function of k looks constant as seen with the roughly equal shading at each vertical.  
runtime_plot <- ggplot(runtime_dataframe, aes(x=n, y=k,col=runtime)) +
  geom_tile(aes(fill=runtime)) +
  labs(title = "Runtime plotted with k and n", x = "n (as a proportion of total dataset rows)")
runtime_plot


# Plot 2 shows the steady increase in runtime as n increases. This appears to be true for all cases of k.
runtime_plot2 <- ggplot(runtime_dataframe, aes(x=n, y = runtime, group=k, col = k)) +
  geom_line() +
  labs(title = "Runtime with Varying n on x-axis", x = "n (as a proportion of total dataset rows)")
runtime_plot2


# Plot 3 shows runtime stays generally pretty constant as k increases.
# Some lines are higher than others due to the increased n. 
runtime_plot3 <- ggplot(runtime_dataframe, aes(x=k, y = runtime, group=n, col = n)) +
  geom_line() +
  labs(title = "Runtime with Varying k on x-axis", x = "k")
runtime_plot3

save(runtime_dataframe, file="data.Rda")


# Most Useful Plot
ggsave(filename="timothy_lee.png", width=16, height = 9) 


# Optional Plots for Further Information
ggsave(filename="timothy_lee2.png", width=16, height = 9)
ggsave(filename="timothy_lee3.png", width=16, height = 9)




# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set: as n increases, runtime increases linearly

# -k: number of neighbors to consider: as k increases, runtime stays the same, so it's constant across increasing k values
# However,the horizontal lines of k are stacked on top of each other (`timothy_lee3.png`), signifying that increasing n
# will increase the runtime in a linear fashion. 

# -d: number of predictors used? In this case d is fixed at 3 but should exhibit same behavior as the k description above


# Based on the above logic, I would write the Big-O runtime algorithmic complexity as
# f(n) = O(nk + nd)

