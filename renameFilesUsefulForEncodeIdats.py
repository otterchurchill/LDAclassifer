#to run: python3.5 renameFilesUsefulForEncodeIdats.py <metadata file> <path of files>

import sys
import os

def main():
    metadata = sys.argv[1]
    path = sys.argv[2] 

    with open(metadata) as metadataFile, open("Output.txt", 'w+') as outfile:
        print("1")
        for line in metadataFile:
            line = line.split()
            typeOfSignal = line[2]
            formatFile = ".idat"
            experimentAcc = line[3]
            fileAcc = line[0]

            fileToLookFor = path + fileAcc + ".idat" 
            print(fileToLookFor)
            if os.path.isfile(path + fileAcc + ".idat"):
                newName = "{0}_{1}{2}{3}".format(experimentAcc, "", typeOfSignal, formatFile)
                outfile.write("{0}\t{1}\n".format(fileAcc, newName))

                os.rename(os.path.join(path,fileAcc + ".idat"), os.path.join(path,newName))

if __name__ == "__main__":
    main()
