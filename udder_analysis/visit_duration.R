library(dplyr)
library(tidyr)
library(lmerTest)

# directories 
# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "lactation_features.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

plot(y = df$notmilk_visit_duration_mean,  x =df$eu_back)

plot(y = df$notmilk_visit_duration_mean,  x =df$eu_front)
     
plot(y = df$notmilk_visit_duration_mean,  x =df$eu_left)

plot(y = df$notmilk_visit_duration_mean,  x =df$eu_right)

hist(df$notmilk_visit_duration_mean)


plot(y = df$notmilk_visit_duration_mean,  x =df$circ_udder)

plot(y = df$notmilk_visit_duration_mean,  x =df$exc_udder)

plot(y = df$notmilk_visit_duration_mean,  x =df$area_udder)

plot(y = df$notmilk_visit_duration_mean,  x =df$peri_udder)

plot(y = df$notmilk_visit_duration_mean,  x =df$yield_visit_mean)
