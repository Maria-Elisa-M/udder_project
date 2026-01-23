library(dplyr)
library(tidyr)
library(lmerTest)
library(ggplot2)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/paper/revision"
setwd(wd)
data_dir <- file.path(wd, "data", fsep = .Platform$file.sep)

# read data
file_name <- "visit_ft_long.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)
df <- df%>%mutate(parity = ifelse(lactation_number > 1, 1, 0), 
                  location = ifelse((teat == "lf")|(teat == "rf"), "front", "rear"), 
                  side = ifelse((teat == "lf")|(teat == "lr"), "left", "right"))

# center and standardize values
cols1 <- c("yield", "interval_sec", "vol", "days_in_milk",  "sarea", "area", "circ", "exc", "peri", "conductivity", "peak_flow", "mean_flow", "len")
df[cols1] <- scale(df[cols1])

# set as factors
cols2 <- c("cow", "teat", "parity", "location", "side")
df <- df%>%mutate_at(cols2, factor)

# drop rows with missing values
vars = c(cols1, cols2)
dfm <- df%>%select(all_of(vars))%>%drop_na()

# visit quarter yield
model_yield <- lmer(yield ~ interval_sec + len + area + days_in_milk + parity + circ + exc + location + side + (1|cow),  data = dfm)
summary(model_yield)
resid <- residuals(model_yield)

png(filename = "model/qqplot_yield.png", width = 800, height = 600)
qqnorm(resid)
dev.off()

png(filename = "model/rfplot_yield.png", width = 800, height = 600)
plot(model_yield, which = 1)
dev.off()

# visit quarter electical conductivity
model_ec <- lmer(conductivity ~ yield + interval_sec + len + area + days_in_milk + parity + circ + exc + location + side +(1|cow),  data = dfm)
summary(model_ec)
resid <- residuals(model_ec)

png(filename = "model/qqplot_ec.png", width = 800, height = 600)
qqnorm(resid)
dev.off()

png(filename = "model/rfplot_ec.png", width = 800, height = 600)
plot(model_ec, which = 1)
dev.off()

# model quarter peak flow
model_pf <- lmer(peak_flow ~ yield + interval_sec + len + area + days_in_milk + parity + circ + exc + location + side + (1|cow),  data = dfm)
summary(model_pf)
resid <- residuals(model_pf)

png(filename = "model/qqplot_pf.png", width = 800, height = 600)
qqnorm(resid)
dev.off()

png(filename = "model/rfplot_pf.png", width = 800, height = 600)
plot(model_pf, which = 1)
dev.off()

# model quarter mean flow
model_mf <- lmer(mean_flow ~ yield + interval_sec +len+ area + days_in_milk + parity + circ + exc + location + side + (1|cow),  data = dfm)
summary(model_mf)
resid <- residuals(model_mf)

png(filename = "model/qqplot_mf.png", width = 800, height = 600)
qqnorm(resid)
dev.off()

png(filename = "model/rfplot_mf.png", width = 800, height = 600)
plot(model_mf, which = 1)
dev.off()

# save the coeffients in a table
coef_yield <- data.frame(coef(summary(as(model_yield,"lmerModLmerTest")))[, 1])
serror_yield <- data.frame(coef(summary(as(model_yield,"lmerModLmerTest")))[, 2])
pv_yield <- data.frame(coef(summary(as(model_yield,"lmerModLmerTest")))[, 5])
colnames(coef_yield) <- "yield"
colnames(pv_yield) <- "yield"
colnames(serror_yield) <- "yield"

coef_pf <- data.frame(coef(summary(as(model_pf,"lmerModLmerTest")))[, 1])
serror_pf <- data.frame(coef(summary(as(model_pf,"lmerModLmerTest")))[, 2])
pv_pf <- data.frame(coef(summary(as(model_pf,"lmerModLmerTest")))[, 5])
colnames(coef_pf) <- "peak_flow"
colnames(pv_pf) <- "peak_flow"
colnames(serror_pf) <- "peak_flow"

coef_ec <- data.frame(coef(summary(as(model_ec,"lmerModLmerTest")))[, 1])
serror_ec <- data.frame(coef(summary(as(model_ec,"lmerModLmerTest")))[, 2])
pv_ec <- data.frame(coef(summary(as(model_ec,"lmerModLmerTest")))[, 5])
colnames(coef_ec) <- "conductivity"
colnames(pv_ec) <- "conductivity"
colnames(serror_ec) <- "conductivity"

coef_mf <- data.frame(coef(summary(as(model_mf,"lmerModLmerTest")))[, 1])
serror_mf <- data.frame(coef(summary(as(model_mf,"lmerModLmerTest")))[, 2])
pv_mf <- data.frame(coef(summary(as(model_mf,"lmerModLmerTest")))[, 5])
colnames(coef_mf) <- "mean_flow"
colnames(pv_mf) <- "mean_flow"
colnames(serror_mf) <- "mean_flow"

merged_coef <- merge(coef_yield, coef_ec, by = 0, all=TRUE)
merged_coef <- merge(merged_coef, coef_mf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_coef <- merge(merged_coef, coef_pf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_coef[, 2:5] <- round(merged_coef[, 2:5], 2)

merged_pv <- merge(pv_yield, pv_ec, by = 0, all=TRUE)
merged_pv <- merge(merged_pv, pv_mf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_pv <- merge(merged_pv, pv_pf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_pv[, 2:5] <- round(merged_pv[, 2:5], 2)

merged_error <- merge(serror_yield, serror_ec, by = 0, all=TRUE)
merged_error <- merge(merged_error, serror_mf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_error <- merge(merged_error, serror_mf, by.x = 'Row.names', by.y = 0, all=TRUE)
merged_error[, 2:5] <- round(merged_error[, 2:5], 2)

write.csv(merged_coef, "C:\\Users\\marie\\rep_codes\\udder_project\\paper\\revision\\tables\\merged_coef.csv", row.names = FALSE)
write.csv(merged_pv, "C:\\Users\\marie\\rep_codes\\udder_project\\paper\\revision\\tables\\merged_pv.csv", row.names = FALSE)
write.csv(merged_error, "C:\\Users\\marie\\rep_codes\\udder_project\\paper\\revision\\tables\\merged_error.csv", row.names = FALSE)
