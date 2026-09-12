library(readr)
library(MASS)
library(pscl)
library(Seurat)
library(dplyr)

# --- load ---
adata_rna <- readRDS("/proj/hyejunglab/cropseq/Alejandro/ND_CROP-Seq/Miscellaneous_Analysis/Outputs/cropseq_seurat_glutamatergic_with_gRNAs.rds")
covariates_df <- read.csv("/proj/hyejunglab/cropseq/Alejandro/Mint/DEG_Analysis/Results/DEG_Analysis/DEG_covariates.csv")
formular_nb <- as.formula(
  "expression ~ group + log(response_n_nonzero) + log(grna_n_nonzero) + log(grna_n_umis) + percent_mt + inlet"
)
expr_mat <- GetAssayData(adata_rna, slot = "counts")

gene_name = "ENSG00000238009"
grna_target = "FOXP2"

y <- as.numeric(expr_mat[gene_name, ])
df0 <- covariates_df
df0$expression <- y

df_sub <- df0 %>%
  filter(grepl(paste0("non-targeting|", grna_target), gRNA_label)) %>%
  mutate(
    group = if_else(grepl(grna_target, gRNA_label), "Perturbed", "Control"),
    group = factor(group, levels = c("Control", "Perturbed"))
  )

rm(df0)


fit <- glm.nb(formular_nb, data = df_sub)
summary(fit)

fit <- zeroinfl(formular_nb, data = df_sub,dist="negbin")
summary(fit)
