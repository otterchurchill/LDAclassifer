# To run python3.5 FileManifest (.#)precentsplit numTests numNonRand numRand > StepInfo

import sys
import pandas as pd
import matplotlib.patches as mpatches
import gc

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import stats
from sklearn.utils import shuffle

#********************************************************************************
def checkArgs(argv):
    if len(argv) != 6 :
        print("toRun: python3.5 path/LDAClassifier.py <Manifest.txt> <ratioToSplit> <NumberOfTests>"
        + "<numNonRand> <numRand>")
        sys.exit()
    try:
        if float(argv[2]) < 0 or float(argv[2]) > 1:
            print("Please give a valid split decimal number between 0 and 1") 
            sys.exit()
    except:
        print("Second argument is not a float!")

    if (int(argv[4]) + int(argv[5])) != int(argv[3]):
        print("The number of non-randomized test plus randomized tests should equal the number of"
        + "tests, so the argument 3 should equal to the sum of argument 4 and 5")

#********************************************************************************
class PDWrapper:
    betaData = None
    typeOf = None
    columnHead = None
    useAsList = []
    specificTrainRatioList = []
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
def getOrder(orderOfUse, OrderOfSpecificTrainRatio , sampleSetName, numOfTests):
    checkOrder = orderOfUse.strip('[]').split(',')
    
    print("checkOrder:", checkOrder)
    if len(checkOrder) < numOfTests:
        print("ERROR:", sampleSetName, "does not have a valid useAsList")
        print("Please remember it has to have " + str(numOfTests) + " to work")
        sys.exit()
    acceptable = {"NA", "TR", "TE", "TR-TE-SP", "TR-SP"}
    
    if not acceptable.issuperset(set(checkOrder)):
        print("ERROR:", sampleSetName, "does not have a valid useAsList ")
        print("please remember it can only contain the strings ['NA', 'TR', 'TE', 'TR-TE-SP', 'TR-SP'] to work")
        sys.exit()
    
    try:
        checkOrderSpecificRatio = [float(i) for i in OrderOfSpecificTrainRatio.strip('[]').split(',')]
        
    
    except:
        print("ERROR:", sampleSetName, "does not have a valid specificTrainRatioList")
        print("please remember it can only contain the strings ['inf',decimal between 1 and 0 such as '.5'] to work")
        sys.exit()
    
    print ("checkOrderSpecificRatio:", checkOrderSpecificRatio) 
    
    for i in checkOrderSpecificRatio:
        if not ((i > 0 and i < 1) or ( i == float("inf"))) :
            raise ValueError("Need specificTrainRatioList to have values between 1 and 0 such as '.5'")

    print("checkOrderSpecificRatio:", checkOrderSpecificRatio)
    if len(checkOrderSpecificRatio) < numOfTests:
        print("ERROR:", sampleSetName, "does not have a valid specificTrainRatioList")
        print("Please remember it has to have " + str(numOfTests) + " to work")
        sys.exit()
    


    return checkOrder, checkOrderSpecificRatio

#********************************************************************************
def isPD(objectToCheck, name):
    if type(objectToCheck) != "pandas.core.frame.DataFrame":
        print("False",name, "is a", type(objectToCheck))
    
#********************************************************************************
def split(precentNeeded,sampleSetList,numOfTests, outfile):
    test = pd.DataFrame()
    train = pd.DataFrame()
    classifTest = pd.DataFrame()
    classifTrain = pd.DataFrame()
   
    for x,sampleSet in enumerate(sampleSetList):
        if sampleSet.useAsList[numOfTests] == "NA":
            continue
        elif sampleSet.useAsList[numOfTests] == "TR":
            train = train.append(sampleSet.betaData)
            classifTrain = classifTrain.append(sampleSet.typeOf)
            print("classifTrain\n", classifTrain)
            print("#OfSamplesFor train sampleSet" + str(x) + ':' + str(len(sampleSet.betaData.index)) + 
            " First element: " + str(sampleSet.typeOf.iloc[0,0]), file=outfile) 
        
        elif sampleSet.useAsList[numOfTests] == "TE":
            test = test.append(sampleSet.betaData)
            classifTest = classifTest.append(sampleSet.typeOf)
            print("#OfSamplesFor test sampleSet" + str(x) + ':' + str(len(sampleSet.betaData.index)) + 
            " First element: " + str(sampleSet.typeOf.iloc[0,0]), file=outfile) 
        
        ### Split 
        elif (sampleSet.useAsList[numOfTests] == "TR-TE-SP") or (sampleSet.useAsList[numOfTests] == "TR-SP"):
            if sampleSet.specificTrainRatioList[numOfTests] == float("inf"):
                trainNum, testNum = getTrainTest(sampleSet.specificTrainRatioList[numOfTests], len(sampleSet.betaData.index))
            else:
                trainNum, testNum = getTrainTest(precentNeeded, len(sampleSet.betaData.index))
                

            print("trainSample"+ str(x), trainNum, "testSample"+ str(x), testNum)  
            #Get betaData train and test          
            sampleTrain = sampleSet.betaData.loc[ : trainNum - 1, :]
            sampleTest = sampleSet.betaData.loc[trainNum :, : ]
            #Get classif data train and test
            cellTrain = sampleSet.typeOf.loc[ : trainNum - 1]
            cellTest = sampleSet.typeOf.loc[ trainNum : ]
            #add the created Training set of beta data and classifs to all training sets
            train = train.append(sampleTrain)
            classifTrain = classifTrain.append(cellTrain)
            print("#OfSamplesFor train sampleSet" + str(x) + ':' + str(len(cellTrain.index)) + 
            " First element: " + str(cellTrain.iloc[0,0]), file=outfile) 

            
            #If its a Train-Test Split will append to test also
            if sampleSet.useAsList[numOfTests] == "TR-TE-SP": 
                test = test.append(sampleTest)
                classifTest = classifTest.append(cellTest)
                print("#OfSamplesFor test sampleSet" + str(x) + ':' + str(len(cellTest.index)) + 
                " First element: " + str(cellTest.iloc[0,0]), file=outfile) 
    
            
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
    print( "Accuracy:", correctPredictions,'\n', file=outfile)
        
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
    checkArgs(sys.argv)    
    fileListFile = sys.argv[1]
    percent = sys.argv[2]
    numOfTests = int(sys.argv[3])
    numOfNonRandTests = int(sys.argv[4])
    numOfRandTests = int(sys.argv[5])

    percent = float(percent)
    print("Split Ratio is", percent)
     
    inPDList = []

    with open (fileListFile, 'r') as inFiles:
        #expecting "BetaFile\tCellTypeManifest"
        for x,sampleLine in enumerate(inFiles):
            
            sampleLine = sampleLine.split()
            
            newInput = PDWrapper()
            
            newInput.betaData, newInput.columnHead = makePD(sampleLine[0], True) 
            newInput.typeOf = makePD(sampleLine[1])
            
            newInput.useAsList, newInput.specificTrainRatioList = getOrder(sampleLine[2],
            sampleLine[3], "sampleset" + str(x), numOfTests) 

            print("BetaValues Sample" +str(x) + "\n", newInput.betaData)
            
            print("ColValues Sample" +str(x) + "\n", newInput.columnHead)
            print("CellType Sample" +str(x) + "\n", newInput.typeOf)
            newInput.pair = x
            
            #check that the patients are in the same order
            xUID = newInput.betaData.iloc[:, 0]
            yUID = newInput.typeOf.iloc[:, 0]

            isSortedFiles(xUID, yUID, "Sample"+ str(x))
            #reduce DTs to relevant information
            newInput.betaData = newInput.betaData.iloc[:, 1:]
            newInput.typeOf = newInput.typeOf.iloc[:, 1:]
            
            
            inPDList.append(newInput)
        
    
    
    isSorted,intersect = checkHeaders(inPDList)
    
    if not isSorted:
        filterBetaData(inPDList,intersect)
    
    outfile =  open('LDA_Results.txt', 'w+')
    
    for testTrialNum in range(0,numOfTests): 
        #****************Non-Randomized Trial**************
        if testTrialNum < numOfNonRandTests: 
            print("TEST #"+ str(testTrialNum))

            print("TEST #"+ str(testTrialNum), file=outfile)
            train, test, trainClassifs, testClassifs = split(percent, inPDList, testTrialNum, outfile)

            #return train, test, classifTrain, classifTest
            print("test\n", test)
            print("train\n", train)
            print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
            print("classifTrain\n", trainClassifs)
            prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
            getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)
            

        if testTrialNum >= numOfNonRandTests:
            #****************Randomized Trail**************
            print("TEST #"+ str(testTrialNum))
            print("TEST #" + str(testTrialNum), file=outfile)

            for i in range (0,10):
                randomize(inPDList, i)
                train, test, trainClassifs, testClassifs = split(percent, inPDList, testTrialNum, outfile)
                    
                #return train, test, classifTrain, classifTest
                print("test\n", test)
                print("train\n", train)
                print("classifTest\n", testClassifs.iloc[:,0 ].tolist())
                print("classifTrain\n", trainClassifs)
                prediction = predict(train, test, trainClassifs, inPDList[0].columnHead)
                getAccuracy(testClassifs.iloc[:, 0 ].tolist(), prediction, outfile)


if __name__ == "__main__":
    main()

