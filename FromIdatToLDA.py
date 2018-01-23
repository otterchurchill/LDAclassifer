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

    outfile = open('PyManifest.txt', 'w+')
    if (whichPart == 'R') or (whichPart == "Both"):

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

                
                print(path + line[1] + "OutputBetaNoXYTran.txt", line[0], line[2], line[3], file=outfile)
                
                sh.cd(startPath) 
    '''
    checkerfile = open('LDACheckerResults', 'w+')
    if (whichPart == "python") or (whichPart == "Both"):
        sh.cd(currentPath) 
        
        pythonCmd("../UsedToPredictClassif/LDAClassifier.py",  "PyManifest.txt", .25,
        _out=checkerfile)

    '''
            

        
    
    


if __name__ == "__main__":
    main()
