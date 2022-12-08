library(tidyverse)
library(data.table)
library(rstatix)

main_auc <- fread("secondary-screen-dose-response-curve-parameters.csv")
chr_8q <- fread("8q_Arm_level_CNAs.csv")
tumor_id <- names(fread("chr8q_affected_melanoma_ovarian_mpnst.txt"))

# choose NF patients with high risk
chr_8q %>% filter(`DepMap ID` %in% tumor_id) -> chr_8q
chr_8q %>% filter(`8q  Arm-level CNAs` == 1) -> gain
chr_8q %>% filter(`8q  Arm-level CNAs` != 1) -> no_gain

# defined chr8q gained groups
gain$`DepMap ID` -> gain_ID
no_gain$`DepMap ID` -> no_gain_ID

main_auc %>% filter(depmap_id %in% gain_ID) %>% select(depmap_id, auc, name) -> gain_auc
main_auc %>% filter(depmap_id %in% no_gain_ID) %>% select(depmap_id, auc, name) -> no_gain_auc
drugs <- unique(gain_auc$name)

# loop through each drug and get p-val
p_val <- c()
median_val <- c()
for (i in 1:length(drugs)){
  gain_auc %>% filter(name == drugs[i]) -> tmp_gain
  no_gain_auc %>% filter(name == drugs[i]) -> tmp_no_gain
  tmp <- wilcox.test(tmp_gain$auc, tmp_no_gain$auc, paired = F, p.adjust.method = "fdr")
  median_diff <- median(tmp_gain$auc)-median(tmp_no_gain$auc)
  p_val <- append(p_val, tmp$p.value)
  median_val <- append(median_val, median_diff)
}

volcano_input <- data.table(name = drugs, log10_adj_p_val = -log10(p_val), median_diff = median_val)
# fwrite(volcano_input, "new_chr8q_wilcoxon_volcano_input.csv")
volcano_input <- fread("new_chr8q_wilcoxon_volcano_input.csv") # use this to save time later you load in the data

#########################################
#  Visualization
# (1) add color for fdr < 0.05 threshold > -log10(0.05); y-axis can stay as log10_adj_p_val
volcano_input %>% mutate(Group = case_when(log10_adj_p_val > -log10(0.05) & median_diff < 0 ~ "chr8q_gain_effective_drug",
                                           log10_adj_p_val > -log10(0.05) & median_diff > 0 ~ "chr8q_no_gain_effective_drug",
                                           log10_adj_p_val <= -log10(0.05) ~ "not_different")) -> volcano_input

# volcano plot
ggplot(data=volcano_input, aes(x=median_diff, y=log10_adj_p_val)) +
  geom_point(aes(color = Group), size = 2) +
  scale_color_manual(values = c("dodgerblue3", "firebrick3", "gray50")) +
  xlab(expression("Effect Size (Median AUC difference between chr8q gain vs chr8q no-gain)")) + 
  ylab(expression("-log"[10]*"adj.p.val")) + 
  guides(color=guide_legend(title=expression("FDR cutoff < -log"[10]*"(0.05)")))+
  theme(text = element_text(size = 20))  

# to bar effect
volcano_input$direction <- ifelse(volcano_input$median_diff < 0,-1,1)

volcano_input <- volcano_input %>% mutate(new_p_val = log10_adj_p_val * direction)
barplot_input <- volcano_input

barplot_input %>% filter(dense_rank(new_p_val) <= 10 | dense_rank(desc(new_p_val)) <= 10) %>% 
  arrange(desc(new_p_val)) -> barplot_input

# barplot to rank significantly different AUC based on Chr8q
ggplot(barplot_input, aes(x = reorder(name, new_p_val), y = new_p_val))+
  geom_col(aes(fill = factor(direction, levels = c(1,-1)))) + 
  coord_flip() +
  xlab(expression("Drug name")) + 
  ylab(expression("Signed.log"[10]*"Adj.p.val")) +
  scale_fill_discrete(labels = c("works better on patients with no chr8 gain",
                                 "works better on patients with chr8 gain")) +
  labs(fill='') +
  theme(text = element_text(size = 20))  

# save tables
volcano_input %>% filter(Group == "chr8q_gain_effective_drug") %>%
  arrange(desc(log10_adj_p_val)) %>% select(name, log10_adj_p_val, median_diff) -> chr8q_gain_effective_drug_table

fwrite(chr8q_gain_effective_drug_table, file = "chr8q_gain_effective_drug_table.csv")

volcano_input %>% filter(Group == "chr8q_no_gain_effective_drug") %>%
  arrange(desc(log10_adj_p_val)) %>% select(name, log10_adj_p_val, median_diff) -> chr8q_no_gain_effective_drug

fwrite(chr8q_no_gain_effective_drug, file = "chr8q_no_gain_effective_drug_table.csv")
#########################################

