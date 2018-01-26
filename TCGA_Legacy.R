#Use to get raw idats from TCGA
#This will produce a Manifest for record keeping. Be careful of duplicate patients so always check
#manifest patients with sort -u, handle duplicates according to Dr.Han.  

library("TCGAbiolinks")

query <- GDCquery(project = "TCGA-STAD",
                  data.category = "Raw microarray data",data.type = "Raw intensities", 
                  experimental.strategy = "Methylation array", legacy = TRUE,
                  file.type = ".idat", platform = "Illumina Human Methylation 450")


query <- GDCquery(project = "TCGA-LIHC", data.category = "Raw microarray data",data.type = "Raw intensities", experimental.strategy = "Methylation array", legacy = TRUE, file.type = ".idat", platform = "Illumina Human Methylation 450")

query <- GDCquery(project = "TCGA-COAD", data.category = "Raw microarray data",data.type = "Raw intensities", experimental.strategy = "Methylation array", legacy = TRUE, file.type = ".idat", platform = "Illumina Human Methylation 450")

query <- GDCquery(project = "TCGA-UCEC", data.category = "Raw microarray data",data.type = "Raw intensities", experimental.strategy = "Methylation array", legacy = TRUE, file.type = ".idat", platform = "Illumina Human Methylation 450")


trycatcher<- function(x)
{
        tryCatch({x},
        error=function(e){cat("ERROR :", conditionMessage(e), "\n")
        cnd<- conditionMessage(e)
        return(cnd)})
}


for ( i in 1:10){
	trycatcher(GDCdownload(query, method="api",files.per.chunk = 20)) } 


match.file.cases <- getResults(query,cols=c("cases","file_name"))
write.table(match.file.cases, "Manifest.txt",sep="\t")

