library(stats)
library(ggplot2)

# directories 
wd <- "C:/Users/marie/rep_codes/udder_project/udder_analysis"
data_dir <- file.path(wd, "long_format_df", fsep = .Platform$file.sep)

# read data
file_name <- "ko_ft_long_side.csv"
file_path <- file.path(data_dir, file_name, fsep = .Platform$file.sep)
df <- read.csv(file_path)

ko_total <- df%>%group_by(cow, side)%>%
  summarise(ko_total = sum(ko), 
            nvisits = n(),
            eu_dist = max(eu), 
            gd_dist = max(gd),
            dim = max(days_in_milk, na.rm = TRUE))

hist(ko_total$ko_total)

plot(ko_total$ko_total, ko_total$eu_dist)

back <- ko_total%>%filter(side == "back")
plot(back$ko_total ~ back$eu_dist)

back <- ko_total%>%filter(side == "back" & ko_total <20)
plot(back$ko_total ~ back$eu_dist)

front <- ko_total%>%filter(side == "front"& ko_total <8)
plot(front$ko_total ~ front$eu_dist)
