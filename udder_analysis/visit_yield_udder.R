library(dplyr)
library(lmerTest)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long_udder.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)


# linear models

model <- lm(yield ~ interval_sec + vol_udder + days_in_milk + lactation_number + sarea_udder + area_udder +
              vol_udder*interval_sec + vol_udder*days_in_milk + lactation_number*days_in_milk, data = df)

summary(model)

model <- lm(yield ~ interval_sec + vol_udder + days_in_milk + lactation_number + sarea_udder, data = df)

summary(model)

model <- lm(vol_udder ~ interval_sec + days_in_milk + lactation_number + yield + 
               lactation_number*days_in_milk, data = df)

summary(model)
