#Script returns Beta Values, with and without Sex Chromosomes, transposed and undtransposed
#To Run put into folder that contains Manifest and idats.  
#/usr/bin/R <ManifestFileName> <OutputFileNameToAppend>
#R version 3.4.1 (2017-06-30) -- "Single Candle"


suppressMessages(library("minfi"))
suppressMessages(library("minfiData"))
suppressMessages(library("IlluminaHumanMethylation450kmanifest"))

args <- commandArgs(trailingOnly=TRUE)
manifest <- args[1]
outAppend <- args[2]

print(manifest)


#Get Experiment accession 
epicTargets <- read.csv(manifest, header=TRUE, sep="\t")

print(epicTargets)

#Create RGchannelSet objects to be able to add PhenoData before read.metharray
epicRgset <- RGChannelSet()


#Creating seperate arrays for Epic and 450k, require experiment_accession_Grn.idat and
#experiment_acc.Red.idat
epicRgset <- read.metharray(epicTargets$Basename, verbose = TRUE, force=TRUE)

fourFiftyKRgset <- convertArray(epicRgset, outType="IlluminaHumanMethylation450k", verbose=TRUE)

#Funnorm chosen for its analysis is to compare different tissues, cell types, or cancers versus
#normals (or any other dataset with global methylation differences
funnorm <- preprocessFunnorm(fourFiftyKRgset)

Beta <- getBeta(funnorm)
outFileName <- paste(outAppend, "OutputBeta.txt", sep='')
write.table(Beta, outFileName, sep="\t")

BetaTran <- t(Beta)
outFileName <- paste(outAppend, "OutputBetaTran.txt", sep='')
write.table(BetaTran, outFileName, sep="\t")


#Get Sex Chromosomes out:  Latest  (doi: 10.12688/f1000research.8839.3) "Filtering"
#Ctrl-F Sex Chrom 

ann450k = getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
keep <- !(featureNames(funnorm) %in% ann450k$Name[ann450k$chr %in% c("chrX","chrY")])
table(keep)
NoXYFunnorm <- funnorm[keep,]

BetaNoXY <- getBeta(NoXYFunnorm)
outFileName <- paste(outAppend, "OutputBetaNoXY.txt", sep='')
write.table(BetaNoXY, outFileName,sep="\t")

BetaNoXYTran <- t(BetaNoXY)
outFileName <- paste(outAppend, "OutputBetaNoXYTran.txt", sep='')
write.table(BetaNoXYTran, outFileName,sep="\t")
