library(dplyr)
library(lmerTest)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

# linear models
df$teat <- as.factor(df$teat)
df$cow <- as.factor(df$cow)

model <- lmer(yield ~ interval_sec + vol + days_in_milk + lactation_number + sarea + area + vol + teat + teat*vol + (1|cow),  data = df)

summary(model)
