import sys
import sh


def main():
    progPath = sys.argv[0]
    whichPart = sys.argv[1]
    fil0 = sys.argv[2]
    progPath= '/'.join(progPath.split('/')[:-1]) + '/'
    print("progPath\t" + progPath)
    

    lines = []
    
    Rcmd = sh.Command("/usr/bin/Rscript")
    pythonCmd = sh.Command("/usr/local/bin/python3.5")
    startPath = str(sh.pwd()).rstrip() + '/'
    startPath= '/'.join(startPath.split('/')[:-1]) + '/'
    print("startPath\t" + startPath)

    
    pythonAndRPath = startPath + progPath

    if (whichPart == 'R') or (whichPart == "Both"):
        outfile = open('PyManifest.txt', 'w+')

        with open (fil0, 'r') as inFil0:
            for line in inFil0:
                line = line.split()

                lines.append(line)

                path= '/'.join(line[0].split('/')[:-1]) + '/'
                print("path\t" + path)

                sh.cd(path)
                
                if line[2] == "450k":
                    Rcmd(pythonAndRPath + "GetBeta450k.R", line[0], line[1])
        
                if line[2] == "Epic":
                    Rcmd(pythonAndRPath + "EpicToBeta450k.R", line[0], line[1])

                
                print(path + line[1] + "OutputBetaNoXYTran.txt", line[0], line[3], line[4], file=outfile)
                
                sh.sed('-i', '1s/^/"PatientID"\t/', path + line[1] + "OutputBetaNoXYTran.txt")
                sh.sed('-i', '1s/^/"PatientID"\t/', path + line[1] + "OutputBetaTran.txt") 
                
                sh.cd(startPath) 




     
    if (whichPart == "python") or (whichPart == "Both"):
        
        if len(sys.argv) != 7 :
                print("toRun: python3.5 path/FromIdatToLDA.py <WhichPart> <Manifest.txt> <ratioToSplit> <NumberOfTests> "
                + "<numNonRand> <numRand>")
                sys.exit()
        
        sh.cd(startPath) 
        checkerfile = open('LDACheckerResults', 'w+')
        
        if whichPart == "Both":
            pythonCmd(pythonAndRPath + "LDAClassifier.py",  "PyManifest.txt", sys.argv[3],
            sys.argv[4], sys.argv[5], sys.argv[6], _out=checkerfile)


        if whichPart == "python":
            pythonCmd(pythonAndRPath + "LDAClassifier.py",  sys.argv[2], sys.argv[3],
            sys.argv[4], sys.argv[5], sys.argv[6], _out=checkerfile)

if __name__ == "__main__":
    main()
