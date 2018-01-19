import sys
import sh

#Please replace the absolute path of the UsedToPredictClassif files, the first argument of Rcmd()
#and pythonCmd

def main():
    whichPart = sys.argv[1]
    fil0 = sys.argv[2]
    

    lines = []
    
    Rcmd = sh.Command("/usr/bin/Rscript")
    pythonCmd = sh.Command("/usr/local/bin/python3.5")
    currentPath = str(sh.pwd()).rstrip() + '/'


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
                    Rcmd("LDAcl/GetBeta450k.R", line[0], line[1])
        
                if line[2] == "Epic":
                    Rcmd("~/MethyWork/UsedToPredictClassif/EpicToBeta450k.R", line[0], line[1])

                
                print(path + line[1] + "OutputBetaNoXY.txt" + '\t' + line[0], file=outfile)
            
    '''
    checkerfile = open('LDACheckerResults', 'w+')
    if (whichPart == "python") or (whichPart == "Both"):
        sh.cd(currentPath) 
        
        pythonCmd("../UsedToPredictClassif/LDAClassifier.py",  "PyManifest.txt", .25,
        _out=checkerfile)

    '''
            

        
    
    


if __name__ == "__main__":
    main()
