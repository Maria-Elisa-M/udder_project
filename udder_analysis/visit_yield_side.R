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
dfo <- df
df <- df%>%mutate(parity = ifelse(lactation_number > 1, 1, 0), 
                  log_int = log(interval_sec), 
                  side = ifelse((teat == 'lf')|(teat == 'rf'), "front", "back"))


ggplot(dfo) + geom_point(aes(y = yield, x = area)) +facet_wrap(~teat, scales = "free")
ggplot(dfo) + geom_point(aes(y = interval_sec, x = vol)) +facet_wrap(~teat, scales = "free")

vms_df <- dfo%>%group_by(device_name, lactation_number)%>%summarise(count =n())

# center and standardize values
cols1 <- c("yield", "interval_sec", "vol", "days_in_milk",  "sarea", "area", "circ", "exc", "peri")
df[cols1] <- scale(df[cols1])

# set as factors
cols2 <- c("cow", "side", "parity", "teat")
df <- df%>%mutate_at(cols2, factor)

# drop rows with missing values
vars = c(cols1, cols2, "lactation_number", "parity", "device_name", "log_int")
dfm <- df%>%select(all_of(vars))%>%drop_na()

# model with individaul effects
model <- lmer(yield ~ interval_sec + vol + days_in_milk + parity + area + circ + exc  + sarea + peri + (1|cow:side),  data = dfm)
summary(model)

MuMIn::r.squaredGLMM(model)

#elimination of non-significant effects
s <- step(model)
final <- get_model(s)
summary(final)

MuMIn::r.squaredGLMM(final)

qqnorm(residuals(final))
qqline(residuals(final))

plot(residuals(final))


# get predictions
new <- tibble(area = dfm$area, side = dfm$side, interval_sec = mean(dfm$interval_sec), lactation_number = 1, cow = dfm$cow[1])
new$pred <- predict(final, new)
dfm$pred <- predict(final, new)
ggplot(dfm) + geom_line(aes(y = pred, x = area)) + geom_point(aes(y = yield, x = area), color = "red") +facet_wrap(~side, scales = "free")
ggplot(dfo) + geom_point(aes(y = yield, x = log(interval_sec)))

# model with intefinal# model with interactions
model <- lmer(yield ~ parity + log_int + area * teat + (1|cow), data = dfm)
summary(model)
s <- step(model)
summary(s)
final <- get_model(s)
summary(final)
MuMIn::r.squaredGLMM(final)

qqnorm(residuals(final))
qqline(residuals(final))

plot(residuals(final))

