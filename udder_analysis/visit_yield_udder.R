library(dplyr)
library(tidyr)
library(stats)
library(ggplot2)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long_udder.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

# check if the data meets the assumptions
# yield
hist(df$yield)
qqnorm(df$yield)
qqline(df$yield)
# volume
hist(df$vol_udder)
qqnorm(df$vol_udder)
qqline(df$vol_udder)

# center and standardize values
cols <- c("yield","sarea_udder", "vol_udder", 
          "days_in_milk", "area_udder",  "peri_udder",
          "exc_udder", "circ_udder","interval_sec")
df[cols] <- scale(df[cols])

# drop rows with missing values
vars = c(cols, "lactation_number")
dfm <- df[vars]%>%drop_na()

#define intercept-only model
intercept_only <- lm(yield ~ 1, data=dfm)
summary(intercept_only)

#define model with all predictors
all <- lm(yield ~ ., data=dfm)
summary(all)

# stepwise regression
forward <- step(intercept_only, direction='forward', scope=formula(all), trace=0)
backward <- step(all, direction='backward', scope=formula(all), trace=0)
both <-step(all, direction='both', scope=formula(all), trace=0)

summary(forward)

# model using all the data
model <- lm(yield ~ interval_sec + lactation_number + vol_udder + 
              peri_udder, data = df)
summary(model)
# residuals







