# To run python3.5 TypeAData.txt TypeACellCalasification.txt TypeBData TypeBClassif (.#)precentsplit outname> StepInfo

import sys
import pandas as pd
import matplotlib.patches as mpatches
import gc

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
def checkHeaders(listOfBetas):
    exemplar = listOfBetas[0].columnHead
    allSorted = True
    for x,pd in enumerate(listOfBetas):
        thisSorted,intersect = isSortedFiles(pd.columnHead, exemplar,"SampleHeader" + str(x))
        
        exemplar = intersect
        if thisSorted == False:
             allSorted = False

    return allSorted, exemplar

#********************************************************************************
def isSortedFiles(listOneUIDs, listTwoUIDs, name):
    if listOneUIDs[1:].equals(listTwoUIDs[1:]):
        print(name, "is ordered identically")
        return True, listOneUIDs

    else:
        '''for listOneI, listTwoI in zip(listOneUIDs,listTwoUIDs):
            print("listOneUIDs:", listOneI)
            print("listTwoUIDs:", listTwoI)
            if listOneI != listTwoI:
                print("something stinks")
        '''
        if "SampleHeader" in name:
            intersect = getIntersect(listOneUIDs, listTwoUIDs)
            #intersect = list(intersect)
            return False, intersect

        if "SampleHeader" not in name:
            print("ERROR:", name, "is not ordered correctly for program to run")
            print("please sort both, ending script now")
            sys.exit()
        

#********************************************************************************
def getIntersect(listOneHeaders, listTwoHeaders):
    listOneHeaders = listOneHeaders.tolist()
    listTwoHeaders = listTwoHeaders.tolist()

    setOne = set(listOneHeaders)
    setTwo = set(listTwoHeaders)

    intersect = setOne.intersection(setTwo)
    
    print("setOne: ", '\n'.join(setOne.difference(setTwo)))
    print("setTwo: ", '\n'.join(setTwo.difference(setOne)))
   
    print("intersect: ", len(intersect))
    intersectDF =  pd.Series(list(intersect))
    return intersectDF
#********************************************************************************
def filterBetaData(ListOfPDWrapper, intersect):
    for x,obj in enumerate(ListOfPDWrapper):
        obj.betaData = obj.betaData.filter(items=(intersect.tolist()))
        print("processed Sample" + str(x) + " :" + str(len(obj.betaData.columns)))
        print(obj.betaData.columns)  


#********************************************************************************
def makePD(name, needColumns=False):
    infile = open(name, 'r')
    
    data = pd.read_csv(infile, sep = '\t')
    columnsHead = data.columns
    columnsHead = columnsHead[1:]

    infile.close()
    
    if needColumns:
        return data, columnsHead
    else:
        print(columnsHead)
        data.rename(columns={columnsHead[0]: "CellTypes"}, inplace=True)
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
def randomize(sampleSetList, state):
    for x,sampleSet in enumerate(sampleSetList):
        print("****************Randomize**************") 
        #concat = pd.concat([sampleSet.typeOf, sampleSet.betaData], axis=1)
        #print("randomize:\n", concat)
        gc.collect()
        rand = sampleSet.betaData.sample(frac=1, random_state=state)
        gc.collect()
        print("randWithindex",rand)
        neededOrder=list(rand.index)
        rand.reset_index(drop = True, inplace = True)
        gc.collect()
        print("rand", rand) 
        sampleSet.betaData = pd.DataFrame(rand)
        
        gc.collect()
        
        sampleSet.typeOf = pd.DataFrame(sampleSet.typeOf.reindex(neededOrder))
        print("TypeOf.index", sampleSet.typeOf)
        
        sampleSet.typeOf.reset_index(drop = True, inplace = True)
        
        print("betaData :", sampleSet.betaData)
        print("typeOf :", sampleSet.typeOf)
        gc.collect()

#********************************************************************************
def getAccuracy(testClassifs, prediction, outfile):
    print (prediction,'\n', sep='')
    print(testClassifs,'\n', sep='')
    
    
    correctPredictionsSummed = 0

    for correct,predict in zip(testClassifs,prediction):
        if correct == predict:
            correctPredictionsSummed += 1
    
    correctPredictions = (correctPredictionsSummed / len(testClassifs))
    print("#OfCorrectPredictions:", correctPredictionsSummed, "#ofPredictions:",len(testClassifs))
    print( "Accuracy:", correctPredictions)
    print("#OfCorrectPredictions:", correctPredictionsSummed, "#ofPredictions:",len(testClassifs),
    file=outfile)
    print( "Accuracy:", correctPredictions, file=outfile)
        
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
            
            print("ColValues Sample" +str(x) + "\n", newInput.columnHead)
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
        
    
    
    isSorted,intersect = checkHeaders(inPDList)
    
    if not isSorted:
        filterBetaData(inPDList,intersect)
    
    outfile =  open('LDA_Results.txt', 'w+')
    
     
    #****************1stTest**************

    print("First Test")
    print("First Test", file=outfile)
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 4)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)
    

    #****************2stTest**************
    print("Second Test")
    print("Second Test", file=outfile)
    for i in range (0,10):
        randomize(inPDList, i)
        train, test, trainClassifs, testClassifs = split(percent, inPDList, 4, True)
            
        #return train, test, classifTrain, classifTest
        print("test\n", test)
        print("train\n", train)
        print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
        print("classifTrain\n", trainClassifs)
        prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
        getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)
    
    
    #****************3rdTest**************

    print("Third Test")
    print("third test", file=outfile)
    train, test, trainClassifs, testClassifs = split(percent, inPDList, 6)

    #return train, test, classifTrain, classifTest
    print("test\n", test)
    print("train\n", train)
    print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
    print("classifTrain\n", trainClassifs)
    prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
    getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)


    #****************4thTest**************

    print("Fourth Test")
    print("Fourth Test", file=outfile)
    for i in range (0,10):
        randomize(inPDList, i)
        train, test, trainClassifs, testClassifs = split(percent, inPDList, 6, True)

        #return train, test, classifTrain, classifTest
        print("test\n", test)
        print("train\n", train)
        print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
        print("classifTrain\n", trainClassifs)
        prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
        getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)
        

if __name__ == "__main__":
    main()

