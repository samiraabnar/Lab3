'''
Created on 25/02/2015

@author: rgalhama
'''

import csv
import sys
import numpy as np
#from src.constants import PATH_TO_PROJECT

#path_data=PATH_TO_PROJECT+"src/data/Frank2011/"
path_data=""
filename=path_data+"FGGT-E{}-data.csv"
filenameC=path_data+"FGGT-E{}-data-CORRECTED-raquel.csv"
FILENAMES={1:filenameC.format(1), 2:filenameC.format(2), 3:filenameC.format(3)}
EXP_COND={1:(1,2,3,4,6,8,12,24), 2:(48,100,300,600,900,1200), 3:(3,4,5,6,9)}
verbose=False

''' Print if verbose mode. '''
def vprint(string):
    if verbose:
        print >> sys.stderr, string
        sys.stderr.flush()
        
        
        
class experimentResults:
    def __init__(self, expId):
        self.expId = expId
        self.TEST_LENGTH=30.0
        self.data = []
        self.performance={}
        self.avg_performance={}
        self.loadData(expId)

    ''' Given an experiment id, it loads the data from the file. It only stores the average performance across subjects for each condition. '''
    def loadData(self, expId):
        filename = FILENAMES[expId]
        f = open(filename, 'Ur')
        reader = csv.reader(f)
        data = []
        row_num = 0
        #Read condition, subject, correct
        for row in reader:
            if row_num > 0: 
                condition = row[1]
                sbjId = row[0]
                correct = row[6]
                data.append((condition, sbjId, correct))
            row_num+=1
        #Performance of each subject
        performance={}
        nsubjects_cond=dict([(k,0.0) for k in EXP_COND[expId]])
        for row in data:
            cond=int(row[0])
            nsubjects_cond[cond] += 1
            if not( (row[0],row[1]) in performance.keys() ):
                performance[(row[0],row[1])] = int(row[2])
            else:
                performance[(row[0],row[1])] += int(row[2])
        nsubjects_cond=dict([(k,(float(v)/self.TEST_LENGTH)) for (k,v) in nsubjects_cond.iteritems()])
        performance = dict(map(lambda (k,v): (k, v/self.TEST_LENGTH), performance.iteritems()))
        vprint("Performance of each human subject: ")
        vprint(performance)
        #Average performance in each condition
        cond_performance = {}
        for k, v in performance.iteritems():
            cond=int(k[0])
            if cond not in cond_performance:
                cond_performance[cond] = [v]
            else:
                cond_performance[cond].append(v)
        avg_performance={}
        std_performance={}
        for cond,v in cond_performance.iteritems():
            avg_performance[cond] = np.mean(np.array(v))
            std_performance[cond] = np.std(np.array(v))
        vprint("Avg performance of each condition: ")
        vprint(avg_performance)
        f.close()
        #Instantiate self
        self.data = data
        self.performance=performance
        self.avg_performance=avg_performance
        self.std_performance=std_performance
        
if __name__=="__main__":
    for idExp in (1,2,3):
        exp = experimentResults(idExp)
        print "Exp ",idExp," avg performance "
        for key in sorted(exp.avg_performance):
            print key, exp.avg_performance[key]
        
#        print "Exp ",idExp," std performance "
#        for key in sorted(exp.std_performance):
#            print key, exp.std_performance[key]

