library(dplyr)
library(lmerTest)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

# center and standardize values
cols1 <- c("yield", "interval_sec", "vol", "days_in_milk",  "sarea", "area", "circ", "exc", "peri")
df[cols1] <- scale(df[cols1])

# set as factors
cols2 <- c("cow", "teat")
df <- df%>%mutate_at(cols2, factor)

# drop rows with missing values
vars = c(cols1, cols2, "lactation_number")
dfm <- df%>%select(all_of(vars))%>%drop_na()

model <- lmer(yield ~ interval_sec + vol + days_in_milk + lactation_number + sarea + area + circ + peri + exc  + teat + (1|cow),  data = dfm)

summary(model)

#elimination of non-significant effects
s <- step(model)

#plot of post-hoc analysis of the final model
plot(s)

summary(s)
