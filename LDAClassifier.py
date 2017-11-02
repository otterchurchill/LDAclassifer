# To run python3.5 TypeAData.txt TypeACellCalasification.txt TypeBData TypeBClassif .... > StepInfo

import sys
import pandas as pd
import matplotlib.patches as mpatches

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats

def isSortedFiles(dataUIDs, typeUIDs, name):
    if dataUIDs.equals(typeUIDs):
        print(name, "is ordered identically")

    else:
        print("ERROR:", name, "is not ordered correctly for program to run")
        print("please sort -k1 on both, ending script now")
        sys.exit()

def makePD(name):
    infile = open(name, 'r')
    
    data = pd.read_csv(infile, sep = '\t')
    data.columns = ["C" + str(i) for i in range(1, len(data.columns)+1)]

    infile.close()

    return data

def isPD(objectToCheck, name):
    if type(objectToCheck) != "pandas.core.frame.DataFrame":
        print("False",name, "is a", type(objectToCheck))
    
def split(precentNeeded,typeA,typeB,typeACell, typeBCell):
    typeATrainNum, typeATestNum = getTrainTest(precentNeeded, len(typeA.index))
    typeBTrainNum, typeBTestNum = getTrainTest(precentNeeded, len(typeB.index))
    
    print("train:", typeATrainNum, "test:", typeATestNum)  

    typeATrain = typeA.loc[ : typeATrainNum - 1, :]
    typeBTrain = typeB.loc[ : typeBTrainNum - 1, : ]
    print("typeATrain: ", typeATrain) 
    typeATest = typeA.loc[typeATrainNum :, : ]
    typeBTest = typeB.loc[typeBTrainNum :, : ]
    print("typeATest: ", typeATest)

    train = typeATrain.append(typeBTrain)
    test = typeATest.append(typeBTest)
    
    cellTrain = typeACell.loc[ : typeATrainNum - 1]
    cellTrain = cellTrain.append(typeBCell.loc[ : typeBTrainNum - 1])

    cellTest = typeACell.loc[ typeATrainNum : ]
    cellTest = cellTest.append(typeBCell.loc[ typeBTrainNum :]) 
    print("cellTest:\n", cellTrain)
    return train, test, cellTrain, cellTest

def getTrainTest(precentNeeded, lenght):
    print("lenght",lenght) 
    trainNum = round(precentNeeded * lenght)
    testNum = lenght - trainNum

    return trainNum, testNum

def predict(train, test , trainClassifs):
    lda = LDA(n_components=2) 
    lda = lda.fit(train, trainClassifs) 
    X_lda = lda.transform(train) 

    YTest = lda.transform(test) 
    YClasifs = lda.predict(test) 
    YClasifsProbability = lda.predict_proba(test) 
    
    print("lda.trans(test):\n", YTest)
    print("lda.predict(test):\n", YClasifs)
    print("predict_prob(test)\n", YClasifsProbability)
    

def main():
    
    typeAfileName = sys.argv[1]
    typeACellFile = sys.argv[2]
    typeBfileName = sys.argv[3]
    typeBCellFile = sys.argv[4]
    percent = sys.argv[5]
    
    percent = float(percent)

    typeAData = makePD(typeAfileName)
    typeBData = makePD(typeBfileName)
    
    typeACellData = makePD(typeACellFile)
    typeBCellData = makePD(typeBCellFile)
    
    print("Step1: Split Ratio is", percent)
    
    print(typeAData)

    print("Cell\n", typeACellData)
    print("Cell\n", typeBCellData)

    xtypeAUID = typeAData.C1
    yTypeACellUID = typeACellData.C1
    
    isSortedFiles(xtypeAUID, yTypeACellUID, "TypeA")
    
    xtypeBUID = typeBData.C1
    yTypeBCellUID = typeBCellData.C1
    
    isSortedFiles(xtypeBUID, yTypeBCellUID, "TypeB")

    xtypeA = typeAData.loc[:, "C2":]
    ytypeA = typeACellData.C2

    xtypeB = typeBData.loc[:, "C2":]
    ytypeB = typeBCellData.C2
    
    train, test, trainClassifs, testClassifs, =split(percent, xtypeA, xtypeB, ytypeA, ytypeB)
    predict(train, test, trainClassifs)

if __name__ == "__main__":
    main()

