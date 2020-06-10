library(ggplot2)
library(dplyr)
library(tidyr)
library(egg)
library(gridExtra)

tableau_data <- read.csv("Documentation/data/Cohort_Summary_data.csv", sep='\t', fileEncoding = "UCS-2LE")

unique_PT_ID <- unique(tableau_data$PT.ID)
unique_PT_ID <- unique_PT_ID[unique_PT_ID != "null"]

diagnosed_table <- tableau_data[tableau_data$Diagnosis != "",]

table_diagnosis <- as.data.frame(table(unique(diagnosed_table[c("PT.ID", "Diagnosis")])$Diagnosis))
table_diagnosis <- table_diagnosis[-1,]

cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",  "#00CC33",  "#FF0033",  "#660099")

mapped_occurences <- tail(sort(table_diagnosis$Freq), 10)
mapped_occurences <- data.frame(Freq=mapped_occurences, color=1:10)
mapped_occurences$pallet <- cbbPalette

table_diagnosis <- merge(table_diagnosis, mapped_occurences, by="Freq", all = TRUE)
table_diagnosis[is.na(table_diagnosis)] <- 0
table_diagnosis$pallet <- ifelse(table_diagnosis$color == 0, "#000000", table_diagnosis$pallet)
table_diagnosis$color <- as.factor(table_diagnosis$color)

top10 <- tail(table_diagnosis, 10)
table_diagnosis <- table_diagnosis[order(table_diagnosis$Var1),]

p<- ggplot(data=table_diagnosis, aes(x=Var1, y=Freq, fill=color)) + 
  geom_col() +
  scale_y_continuous(limits = c(0,max(table_diagnosis$Freq)), expand = c(0, 0)) +
  theme_bw() +
  scale_fill_manual(values=append(cbbPalette, "#000000", 0)) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.text.x = element_text(angle=90, hjust=1, vjust=0.5, colour = table_diagnosis$pallet),
        axis.ticks.x=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)
  
p

p<- ggplot(data=table_diagnosis, aes(x=Freq, y=Var1, fill=color)) + 
  geom_col() +
  scale_x_continuous(limits = c(0,max(table_diagnosis$Freq)), expand = c(0, 0)) +
  theme_bw() +
  scale_fill_manual(values=append(cbbPalette, "#000000", 0)) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.text.y = element_text(colour = table_diagnosis$pallet, size=6),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)

p

# Next figure

data_figure2 <- tableau_data[tableau_data$Diagnosis %in% top10$Var1,]
data_figure2 <- data_figure2 %>% select(Diagnosis, Media.type, Status..Resolution) %>% filter(Media.type != "")
#Freq_table <- data_figure2 %>% count(Media.type, Diagnosis, Status..Resolution, name="n2") %>% group_by(Diagnosis) %>% mutate(prop_verified = prop.table(n2)) %>% filter(Status..Resolution == "Verified Tumor") %>% select(-Status..Resolution, -n2)
#data_figure2 <- data_figure2 %>% select(-Status..Resolution) %>% count(Diagnosis, Media.type) %>% group_by(Diagnosis) %>% mutate(prop = prop.table(n)) %>% mutate(ypos = cumsum(prop)- 0.5*prop ) %>% left_join(Freq_table) %>% replace(is.na(.), 0)
data_figure2 <- data_figure2 %>% select(-Status..Resolution) %>% count(Diagnosis, Media.type) %>% group_by(Diagnosis) %>% mutate(prop = prop.table(n)) %>% mutate(count = sum(n))
p <- ggplot(data_figure2, aes(x="", prop, fill=Media.type)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void() + facet_wrap( ~ Diagnosis, nrow=2) +
tag_facet(p, x = 1, y = 1, hjust = 0.5, vjust=7, tag_pool = unique(data_figure2$count), open = "", close = "")
p
tag_facet(p, x = 1, y = 1, hjust = 0.5, vjust=7, tag_pool = unique(data_figure2$count), open = "", close = "")

# Figures Terra data
spreedsheet_small <- read.csv("Documentation/data/spreadsheet_small.tsv", sep='\t', fileEncoding = "utf-8")

# Panel
data_figure3 <- spreedsheet_small[spreedsheet_small["PANEL_renamed_bai_file"] != "nan",]
#P1
data_figure3_p1 <- as.data.frame(table(data_figure3["PANEL_renamed_bai_file"]))
data_figure3_p1 <- data.frame(type=c("all", "unique"), y=c(sum(data_figure3_p1$Freq), length(data_figure3_p1$Var1)))
p1<- ggplot(data=data_figure3_p1, aes(x=type, y=y)) + 
  geom_col() +
  theme_bw() +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)

#P2
data_figure3_p2 <- as.data.frame(table(data_figure3["Growth"]))
p2 <- ggplot(data=data_figure3_p2, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

data_figure3_p2 <- as.data.frame(table(data_figure3["Growth"]))
p2 <- ggplot(data=data_figure3_p2, aes(x=Var1, y=Freq, fill=Var1)) + 
  geom_col() +
  theme_bw() +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)

#P3
data_figure3_p3 <- as.data.frame(table(data_figure3["Dataset"]))
### Why is this necessary?
data_figure3_p3 <- data_figure3_p3[data_figure3_p3["Var1"] != "",]
p3 <- ggplot(data=data_figure3_p3, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

#P4
data_figure3_p4 <- as.data.frame(table(data_figure3["Media.type"]))
p4 <- ggplot(data=data_figure3_p4, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

#P5
data_figure3_p5 <- as.data.frame(table(data_figure3["Primary.Disease"]))
data_figure3_p5 <- data_figure3_p5[data_figure3_p5["Var1"] != "",]
p5 <- ggplot(data=data_figure3_p5, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

#P6
data_figure3_p6 <- as.data.frame(table(data_figure3["Dimension"]))
data_figure3_p6 <- data_figure3_p6[data_figure3_p6["Var1"] != "",]
p6 <- ggplot(data=data_figure3_p6, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

grid.arrange(p1, p2, p3, ncol=3, nrow=1)

# RNA
data_figure4 <- spreedsheet_small[spreedsheet_small["RNA_fastq1"] != "nan",]

data_figure4_p1 <- as.data.frame(table(data_figure4["RNA_fastq1"]))
data_figure4_p1 <- data.frame(type=c("all", "unique"), y=c(sum(data_figure4_p1$Freq), length(data_figure4_p1$Var1)))
p1<- ggplot(data=data_figure4_p1, aes(x=type, y=y)) + 
  geom_col() +
  theme_bw() +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)

#P2
data_figure4_p2 <- as.data.frame(table(data_figure4["Growth"]))
p2 <- ggplot(data=data_figure4_p2, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

data_figure4_p2 <- as.data.frame(table(data_figure4["Growth"]))
p2 <- ggplot(data=data_figure4_p2, aes(x=Var1, y=Freq, fill=Var1)) + 
  geom_col() +
  theme_bw() +
  theme(panel.grid = element_blank(),
        panel.border = element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),)


#P3
data_figure4_p3 <- as.data.frame(table(data_figure4["Dataset"]))
### Why is this necessary?
data_figure4_p3 <- data_figure4_p3[data_figure4_p3["Var1"] != "",]
p3 <- ggplot(data=data_figure4_p3, aes(x="", y=Freq, fill=Var1)) + geom_bar(stat="identity", width=1) + coord_polar("y", start=0) + theme_void()

grid.arrange(p1, p2, p3, ncol=3, nrow=1)


