if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# create the train and test set from edx.  We will make it 10% of the edx set.
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, 
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

##create the Rmse function for predictions
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Get a baseline Model for predictions
baseline <- mean(train_set$rating)
baseline
## Calculate the RMSE of the baseline training set against the test set 
RMSE(baseline, test_set$rating)
## Create a dataset by taking difference between the rating and the baseline 
movie <- train_set %>% 
  group_by(movieId) %>% 
  summarize(movie = mean(rating - baseline))
## Creating a predictor using our baseline and adding the average movie rating
predicted_ratings <- baseline + test_set %>% 
  left_join(movie, by='movieId') %>%
  pull(movie)
RMSE(predicted_ratings, test_set$rating)

##Create a user data set.  Take the average of the combined rating, baseline, and movie.   
user <- train_set %>% 
  left_join(movie, by='movieId') %>%
  group_by(userId) %>%
  summarize(user = mean(rating - baseline - movie))

##We will add the movie and user data sets to the predictor
predictedRatings <- test_set %>% 
  left_join(movie, by='movieId') %>%
  left_join(user, by='userId') %>%
  mutate(pred = baseline + movie + user) %>%
  pull(pred)
RMSE(predictedRatings, test_set$rating)

## Regularization ##

# lambda is a tuning parameter we can use cross-validation to find the best paramater choose it.
# Doing lambdas 0-10 by .25 increments will find a lambda around 3-6 that will work
# For the final model we will use lambdas 3-5 at intervals of 1.
lambdas <- seq(3, 5, 1)

rmses <- sapply(lambdas, function(l){
  
  baseline <- mean(train_set$rating)
  
  movie <- train_set %>% 
    group_by(movieId) %>%
    summarize(movie = sum(rating - baseline)/(n()+l))
  
  user <- train_set %>% 
    left_join(movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(user = sum(rating - movie - baseline)/(n()+l))
  
  predictedRatings <- 
    train_set %>% 
    left_join(movie, by = "movieId") %>%
    left_join(user, by = "userId") %>%
    mutate(pred = baseline + movie + user) %>%
    pull(pred)
  
  return(RMSE(predictedRatings, train_set$rating))
})

##this graph gives us a visual representation of the plotted rmse with the Lambda tuning parameter
qplot(lambdas, rmses)  
##Finding the lowest RMSE and choosing the lambda
lambda <- lambdas[which.min(rmses)]
lambda

#Final RMSE test - Training the edx set on the final model
movie <- edx %>% 
  group_by(movieId) %>%
  summarize(movie = sum(rating - baseline)/(n()+lambda))

user <- edx %>% 
  left_join(movie, by="movieId") %>%
  group_by(userId) %>%
  summarize(user = sum(rating - movie - baseline)/(n()+lambda))

##Using validation set to create the final predictor

predictedRatings <- validation %>% 
  left_join(movie, by = "movieId") %>%
  left_join(user, by = "userId") %>%
  mutate(pred = baseline + movie + user) %>%
  pull(pred)

##testing the final predictor on the validation set

RMSE(predictedRatings, validation$rating)