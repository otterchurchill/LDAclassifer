# To run python3.5 TypeAData.txt TypeACellCalasification.txt TypeBData TypeBClassif (.#)precentsplit outname> StepInfo

import sys
import pandas as pd
import matplotlib.patches as mpatches

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats
from sklearn.utils import shuffle

#********************************************************************************
class PDWrapper:
    betaData = None
    typeOf = None
    columnHead = None
    pair = -1


#********************************************************************************
def isSortedFiles(listOneUIDs, listTwoUIDs, name):
    if listOneUIDs[1:].equals(listTwoUIDs[1:]):
        print(name, "is ordered identically")

    else:
        for listOneI, listTwoI in zip(listOneUIDs,listTwoUIDs):
            print("listOneUIDs:", listOneI)
            print("listTwoUIDs:", listTwoI)
            if listOneI != listTwoI:
                print("something stinks")


        print("ERROR:", name, "is not ordered correctly for program to run")
        print("please sort both, ending script now")
        sys.exit()


#********************************************************************************
def checkHeaders(listOfBetas):
    exemplar = listOfBetas[0].columnHead
    for x,pd in enumerate(listOfBetas):
        isSortedFiles(pd.columnHead, exemplar,"SampleHeader" + str(x))
        
#********************************************************************************
def makePD(name, needColumns=False):
    infile = open(name, 'r')
    
    data = pd.read_csv(infile, sep = '\t')
    columns = data.columns
    columns = columns[1:]

    infile.close()
    
    if needColumns:
        return data, columns
    else:
        return data

#********************************************************************************
def isPD(objectToCheck, name):
    if type(objectToCheck) != "pandas.core.frame.DataFrame":
        print("False",name, "is a", type(objectToCheck))
    
#********************************************************************************
def split(precentNeeded,sampleList,lenght, lastTwoSplit=False):
    test = pd.DataFrame()
    train = pd.DataFrame()
    classifTest = pd.DataFrame()
    classifTrain = pd.DataFrame()
   
    for x,sample in enumerate(sampleList):
        if x < (lenght-2):
            train = train.append(sample.betaData)
            classifTrain = classifTrain.append(sample.typeOf)
            print("classifTrain\n", classifTrain)

        elif (x >= lenght):
            break
        elif ((x >= (lenght-2)) and not lastTwoSplit):
            test = test.append(sample.betaData)
            classifTest = classifTest.append(sample.typeOf)
        elif ((x >= (lenght-2)) and lastTwoSplit) :

            trainNum, testNum = getTrainTest(precentNeeded, len(sample.betaData.index))

            print("trainSample"+ str(x), trainNum, "testSample"+ str(x), testNum)  
             
    
            sampleTrain = sample.betaData.loc[ : trainNum - 1, :]
            sampleTest = sample.betaData.loc[trainNum :, : ]

            test = test.append(sampleTest)
            train = train.append(sampleTrain)

            cellTrain = sample.typeOf.loc[ : trainNum - 1]
            cellTest = sample.typeOf.loc[ trainNum : ]
    
    
            classifTrain = classifTrain.append(cellTrain)
            classifTest = classifTest.append(cellTest)
    
            print("cellTest:\n", cellTest)
    
    

    return train, test, classifTrain, classifTest

#********************************************************************************
def getTrainTest(precentNeeded, lenght):
    print("lenght",lenght) 
    trainNum = round(precentNeeded * lenght)
    testNum = lenght - trainNum

    return trainNum, testNum

#********************************************************************************
def predict(train, test , trainClassifs, varibles):
    lda = LDA(n_components=2) 
    lda = lda.fit(train, trainClassifs) 
    X_lda = lda.transform(train) 
    
    
    print('{:*^70}'.format('*'))
    print('{:^70}'.format('LinearDiscriminantAnalysis'))
    print('{:*^70}'.format('*'))
    
    scalings(lda, train, varibles, True)
    
    YTest = lda.transform(test) 
    YClasifs = lda.predict(test) 
    YClasifsProbability = pd.DataFrame(lda.predict_proba(test), columns=lda.classes_)
    
    print("lda.trans(test):\n", YTest)
    print("lda.predict(test):\n", YClasifs)
    print("predict_prob(test)\n", YClasifsProbability)
    return YClasifs

#********************************************************************************
def randomize(train, test, trainClassifs, testClassifs, state):
    
    concat = pd.concat([train, trainClassifs], axis=1)
    print("randomize:\n", concat)
    concatRand = concat.sample(frac=1, random_state=state)
    concatRand.reset_index(drop =True)
    return concatRand


#********************************************************************************
def getAccuracy(testClassifs, prediction):
    print (prediction,'\n', sep='')
    print(testClassifs,'\n', sep='')
    
    
    correctPredictionsSummed = 0

    for correct,predict in zip(testClassifs,prediction):
        if correct == predict:
            correctPredictionsSummed += 1
    
    correctPredictions = (correctPredictionsSummed / len(testClassifs))
    print("#OfCorrectPredictions:", correctPredictionsSummed, "#ofPredictions:",len(testClassifs))
    print( "Accuracy:", correctPredictions)
        
#********************************************************************************
def scalings(lda, X, varibles, out=False):
#source:http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html#loadings-for-the-discriminant-functions
    ret = pd.DataFrame(lda.scalings_, index=X.columns, columns=["LD"+str(i+1) for i in range(lda.scalings_.shape[1])])
    varibles = pd.DataFrame( varibles)
    if out:
        pd.set_option('display.max_rows', len(ret))
        pd.options.display.float_format = '{:,.12f}'.format
        print("Coefficients of linear discriminants:")
        retSorted = ret.reindex(ret.LD1.abs().sort_values(ascending=False).index)
        
        #print("SortedLDAVars\n", retSorted, sep='')

    return ret

#********************************************************************************

def main():
    
    fileListFile = sys.argv[1]
    percent = sys.argv[2]
    
    percent = float(percent)
    print("Split Ratio is", percent)
    
    inPDList = []

    with open (fileListFile, 'r') as inFiles:
        #expecting "BetaFile\tCellTypeManifest"
        for x,sampleLine in enumerate(inFiles):
            
            sampleLine = sampleLine.split()
            
            newInput = PDWrapper()
            newInputManifest = PDWrapper()
            
            newInput.betaData, newInput.columnHead = makePD(sampleLine[0], True)
            
            newInput.typeOf = makePD(sampleLine[1])
            print("BetaValues Sample" +str(x) + "\n", newInput.betaData)
            
            print("CellType Sample" +str(x) + "\n", newInput.typeOf)
            newInput.pair = x
            
            inPDList.append(newInput)
            #check that the patients are in the same order
            xUID = newInput.betaData.iloc[:, 0]
            yUID = newInput.typeOf.iloc[:, 0]

            isSortedFiles(xUID, yUID, "Sample"+ str(x))
            #reduce DTs to relevant information
            newInput.betaData = newInput.betaData.iloc[:, 1:]
            newInput.typeOf = newInput.typeOf.iloc[:, 1:]
        
    
    
    checkHeaders(inPDList)
    
    #****************1stTest**************

    print("First Test")
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 4)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction)


    #****************2stTest**************

    print("Second Test")
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 4, True)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction)
    #****************3rdTest**************

    print("Third Test")
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 6)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction)


    #****************4thTest**************

    print("Fourth Test")
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 6, True)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction)
if __name__ == "__main__":
    main()

