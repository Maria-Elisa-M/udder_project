library(dplyr)
library(tidyr)
library(lmerTest)
library(ggplot2)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)
df <- df%>%mutate(parity = ifelse(lactation_number > 1, 1, 0))


ggplot(df) + geom_point(aes(y = yield, x = area)) +facet_grid(vars(teat))

# center and standardize values
cols1 <- c("yield", "interval_sec", "vol", "days_in_milk",  "sarea", "area", "circ", "exc", "peri")
df[cols1] <- scale(df[cols1])

# set as factors
cols2 <- c("cow", "teat", "parity")
df <- df%>%mutate_at(cols2, factor)

# drop rows with missing values
vars = c(cols1, cols2, "lactation_number", "parity")
dfm <- df%>%select(all_of(vars))%>%drop_na()

# model with individaul effects
model <- lmer(yield ~ interval_sec + vol + days_in_milk + lactation_number + sarea + area + circ + peri + exc  + teat + (1|cow),  data = dfm)
summary(model)

MuMIn::r.squaredGLMM(model)

model2 <- lmer(yield ~ interval_sec + vol + days_in_milk + parity + circ + exc  + teat + (1|cow),  data = dfm)
summary(model2)

MuMIn::r.squaredGLMM(model2)

#elimination of non-significant effects
s <- step(model)
final <- get_model(s)
summary(final)

MuMIn::r.squaredGLMM(final)

qqnorm(residuals(final))
qqline(residuals(final))

plot(residuals(final))




# model with intefinal# model with interactions
model <- lmer(yield ~ lactation_number + interval_sec + area * teat + (1|cow), data = dfm)
summary(model)
s <- step(model)
summary(s)
s
MuMIn::r.squaredGLMM(s)


