library(dplyr)
library(tidyr)
library(stats)
library(ggplot2)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "ko_ft_long_teat.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

dfdim <- df%>%group_by(days_in_milk, lactation_number, animal_number)%>%
  summarise(ko = sum(ko_bin))%>%ungroup()%>%
  group_by(days_in_milk, lactation_number)%>%
  summarise(ko = sum(ko), 
            count = n())

ggplot(dfdim) + geom_point(aes(x = days_in_milk, y = ko, color = as.factor(lactation_number)))

ko_total <- df%>%group_by(cow, teat)%>%
  summarise(ko_total = sum(ko_bin), 
            nvisits = n(),
            teat_len = max(len), 
            dim = max(days_in_milk), 
            back_dist)


hist(ko_total$teat_len)
plot(ko_total$ko_total/ko_total$nvisits ~ ko_total$teat_len)



ko_total0 <- df%>%group_by(cow, teat)%>%
  summarise(ko_total = sum(ko_bin), 
            nvisits = n(),
            teat_len = max(len))%>%filter(ko_total ==0)

hist(ko_total0$teat_len)
