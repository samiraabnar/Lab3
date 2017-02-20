#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
    @author: Raquel G. Alhama
    
    Shotgun stochastic hill climbing: iterative stochastic hill climbing with different [random] initial conditions.

u
    Right now it is prepared only Frank et al. 2010 data from Cognition paper.
    This is the script that combines the results of all experiments. 
    It doesn't make much sense to have a separate script but this is faster right now. Should fix it after the deadline (...)

'''

import sys
import os
import copy
import argparse
import numpy as np
import logging
import time
import pickle
import src.RnR as RnR
import src.stimuli.Frank2010 as fr
from src.utils.writer import writerRnR
import src.randGlobal as randGlobal

verbose=False
MAX_MU=1.2
timestamp = time.strftime("%d.%m.%Y.%H:%M")
logFile = "hillClimbing.{0}.log".format(timestamp)
logger = logging.getLogger(logFile)
resultsFile = open("frank2010hC.{0}.results".format(timestamp), "w+")
outputFolder= "RnR-Frank2010-{0}".format(timestamp)
#sys.path.append('/usr/lib64/python2.6/site-packages')  # For laco machines

''' Print if verbose mode. '''
def vprint(string):
    if verbose:
        print >> sys.stderr, string
        sys.stderr.flush()

''' class for hyperparams'''
class weight:

    def __init__(self, minim=0.0, maxim=1.0):
        self.clamped = False
        self.min = minim
        self.max = maxim
        self.value = self.min


    def update(self, new_value):
        if not self.clamped and new_value >= self.min and new_value <= self.max:
            self.value = new_value

    def clamp(self, value):
        self.update(value)
        self.clamped = True
    

''' New weight vector '''
def newWT():
    A = weight(0.0, 1.0)
    B = weight(0.0, 1.0)
    D = weight(0.0, 1.0)
    munp = weight(0.0, MAX_MU)
    muwp = weight(0.0, MAX_MU)
    newWT = {"A":A, "B":B, "D":D, "munp":munp, "muwp":muwp}

    return newWT





''' Simulate one experiment and return resulting performance of the model. Average is computed over actual choices.'''
def simulateAvgChosen(next_wT,  exp, runsModel, nmax):

    performance_runs = dict([(k,[]) for k in exp.CONDS_LIST])
    avg_performance_runs = dict.fromkeys(exp.CONDS_LIST)
    
    for cnd,expcnd in exp.cnd.iteritems():    
        mm = RnR.RnRv3(next_wT["A"].value, next_wT["B"].value, next_wT["D"].value, next_wT["munp"].value, next_wT["muwp"].value, nmax, "variable")
        wr = writerRnR(outputFolder, mm, "EXP"+str(exp.expId)+"COND"+str(expcnd.condition))
        
        
        for run in range(1, runsModel+1):
     
            condition_text="Frank2010, Experiment "+str(exp.expId) +" Condition: " + str(expcnd.condition)
            stream=expcnd.stream
            tic2=time.clock()
            mm.memorizeOnline(stream)
            wr.writeResultsRnRv3(condition_text, stream, run, mm)
            tac2=time.clock()
            vprint("Time memorizing (RnR):"+str(tac2-tic2))
            correct=0
            incorrect=0
            for pair in expcnd.test:
                (prob_x, prob_y, chosen) = mm.Luce(pair[0], pair[1])    
                if chosen in expcnd.words:
                    correct += 1
                else:
                    incorrect += 1
            percentage_correct = correct * 100.0 / (correct + incorrect)
            performance_runs[cnd].append(percentage_correct)
        perf_cond = np.array(performance_runs[cnd])
        avg_performance_runs[cnd] = (np.mean(perf_cond), np.std(perf_cond)) 

    return avg_performance_runs


''' Simulate one experiment and return resulting performance of the model. Average is computed over probability of correct choice (eq. to Frank et al. 2010)'''
def simulate(next_wT, exp, candidates, runsModel, nmax):

    performance_runs = dict([(k,[]) for k in exp.CONDS_LIST])
    avg_performance_runs = dict.fromkeys(exp.CONDS_LIST)

    for cnd,expcnd in exp.cnd.iteritems():
        mm = RnR.RnRv3(next_wT["A"].value, next_wT["B"].value, next_wT["D"].value, next_wT["munp"].value, next_wT["muwp"].value, nmax, "variable")
        wr = writerRnR(outputFolder, mm, "EXP"+str(exp.expId)+"COND"+str(expcnd.condition))


        for run in range(1, runsModel+1):

            condition_text="Frank2010, Experiment "+str(exp.expId) +" Condition: " + str(expcnd.condition)
            #stream="##".join(expcnd.stimuli)
            stream=expcnd.stream
            mm.memorizeCandidates(candidates[exp.expId][cnd])
            wr.writeResultsRnRv3(condition_text, stream, run, mm)
            avgPerf=0
            for pair in expcnd.test:
                (prob_x, prob_y, chosen) = mm.Luce(pair[0], pair[1])
                if pair[0] in expcnd.words:
                    avgPerf+=prob_x
                else:
                    avgPerf+=prob_y
            percentage_correct = avgPerf*100/float(len(expcnd.test))
            performance_runs[cnd].append(percentage_correct)
        perf_cond = np.array(performance_runs[cnd])
        avg_performance_runs[cnd] = (np.mean(perf_cond), np.std(perf_cond))

    return avg_performance_runs

'''Evaluate 1 model'''
def evaluate(metric, responsesModel, experiment):
    
    if metric == "pearsonr":
        next_eval = experiment.pearsonR(responsesModel)
    elif metric == "mse":
        next_eval = experiment.mse(responsesModel)
    else:
        raise Exception("Unknown metric: ", metric) 
    vprint(metric+": "+str(next_eval))
    
    return next_eval


    

def stochasticHillClimbing(seed, maxIterations, maxJump, experiments, candidates, metric, runsModel, nmax):


    next_wT = newWT()
    best_wT = newWT()
    
    for k, v in seed.iteritems():
        next_wT[k].value = copy.deepcopy(seed[k].value)
        next_wT[k].clamped = copy.deepcopy(seed[k].clamped)
        best_wT[k].value = copy.deepcopy(seed[k].value)
        best_wT[k].clamped = copy.deepcopy(seed[k].clamped)

    initial=-1.0 if args.metric == "pearsonr" else float(sys.maxint)
    best_eval_each={1:initial, 2:initial, 3:initial}
    total_best_eval=sum(best_eval_each.values())
    finalData = {}
    
    for i in range(0, maxIterations):
        
        # Simulate all experiments with weights wT and evaluate
        responsesModels = {}
        for exp in (1,2,3):
            respRnRExp = simulate(next_wT, experiments[exp], candidates, runsModel, nmax)
            responsesModels[exp]=respRnRExp
        vprint(responsesModels)
        
        # Evaluate simulation of each experiment
        next_eval_each={}
        total_next_eval=0
        for exp,respModel in responsesModels.iteritems():
            ne=evaluate(metric, dict([(k,v) for k,(v,_) in respModel.iteritems()]), experiments[exp])
            next_eval_each[exp]=ne
            total_next_eval+=ne
        
        # Write in log 
        logger.debug("Iteration {0}".format(i))
        text = "Weights: A={0},B={1},D={2}".format(next_wT["A"].value, next_wT["B"].value, next_wT["D"].value)
        text += ",munp={0},muwp={1}".format(next_wT["munp"].value, next_wT["muwp"].value) 
        logger.debug(text)
        logger.debug("{0}: {1}".format(metric, next_eval_each))
        logger.debug("Responses (mean,std): " + str(responsesModels))
        #Write in results file 
        resultsFile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(next_wT["A"].value, next_wT["B"].value, 
                                                                                 next_wT["D"].value, next_wT["munp"].value, next_wT["muwp"].value, 
                                                                                 next_eval_each[1], next_eval_each[2], next_eval_each[3], total_next_eval))	
        # Decide if we keep it as actual point
        if better_than(total_next_eval, total_best_eval, args.metric):
            for k, v in next_wT.iteritems():
                best_wT[k].value = next_wT[k].value
                best_wT[k].clamped = next_wT[k].clamped
            total_best_eval = total_next_eval
            best_eval_each = copy.deepcopy(next_eval_each)
            finalData = copy.deepcopy(responsesModels)
            logger.debug("Kept as better point!")

        # Next random neighbour derived from best point		
        for k, v in next_wT.iteritems():
            if not v.clamped:
                var = min(v.max - best_wT[k].value, best_wT[k].value - v.min) * maxJump
                next_wT[k].update(randGlobal.myRandom.gauss(best_wT[k].value, var))
    

    return best_wT, best_eval_each, finalData


def better_than(a, b, metric):
    if metric == "pearsonr":
        return a > b 
    elif metric == "mse":
        return a < b
    else:
        raise Exception("Unexpected metric: ", metric)

def strArgs(args):
    text = "#Arguments of the program:\n"
    for k, v in args.__dict__.iteritems():
        text += "#{0} = {1}\n".format(k, v)
    return text

def strParameters(seed):
    text = ""
    for k,v in seed.iteritems():
        text += "#" + k + " Min:" + str(v.min) + " Max: " + str(v.max) + "\n"
        if v.clamped:
            text += "#" + k + " is clamped to value: " + str(v.value) + "\n"
    return text


''' 
   Prints summary of best simulation
'''
def summary(best_data, best_weights, metric, metric_each, experiments):


    text = "Results of simulation: \n"
    text += str(best_data)
    text += "\nFitted to {0}. Sum for the three experiments: {1}\n".format(metric,sum(metric_each.values()))
    
    text += "Best parameter setting: A={0} B={1} D={2}\n".format(best_weights["A"].value, best_weights["B"].value, best_weights["D"].value)
    text += "                        munp={0} muwp={1}\n".format(best_weights["munp"].value, best_weights["muwp"].value)
    text += "nmax={}".format(args.nmax)

    for idExp,exp in experiments.iteritems():
        text += "\nExperiment " +str(idExp)
        
        text += metric + " " + str(metric_each[idExp])
        
#        text += "Other measures:\n"
#        if metric == "pearsonr":
#            alt_metric = "mse"
#            alt_metric_value = exp.mse(best_data)
#        else:
#            alt_metric = "pearsonr"
#            alt_metric_value = exp.pearsonR(best_data)
#        text += alt_metric + ": " + str(alt_metric_value)

    fh = open("summary." + timestamp + ".txt", "w")
    fh.write(text)
    fh.close()



def main(args):
    # Logging
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logFile)
    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    #Create experiments and stream candidates for rnr
    vprint("Creating an instance of all 3 experiments in Frank et al. 2010...")
    experiments={}
    candidates={}
    for exp in (1,2,3):
        experiments[exp] = fr.experiment(exp)

    if not args.candidatesFile is None and args.candidatesFile != "":
        fh = open(args.candidatesFile,'rb')
        candidates = pickle.load(fh)
        fh.close()
        print "Pickle loaded!"
    else:
        for exp in (1,2,3):
            candidates[exp] = dict.fromkeys(experiments[exp].CONDS_LIST)
            for cnd,expcnd in experiments[exp].cnd.iteritems():
                candidates[exp][cnd] = RnR.RnRv3.createCandidates(expcnd.stream, args.nmax)

        fh=open("candidates_nmax{}.pckl".format(args.nmax), "wb")
        pickle.dump(candidates, fh)
        fh.close()


    # Initial seed with model weights
    A = weight(0.0, 1.0)
    B = weight(0.0, 1.0)
    D = weight(0.0, 1.0)
    munp = weight(0.0, 2.0)
    muwp = weight(0.0, 2.0)
    
    #Not explore munp for the moment:
    #munp.clamp(1.0)


    # Default hardcodable seed (only applies if set to --forceSeed)
    if args.forceSeed:
        #A.update(0.00531775517026)
        #B.update(0.948832354007)
        #D.update(0.827529555613)
        #muwp.update(0.284968373673)

        # A.update(0.00761030007337 )
        # B.update(0.9228192756)
        # D.update(0.865632151688)
        # muwp.update(0.23443846402)
        # munp.update(1.0)

        #Seed for nmax=5
        A.update(0.086)
        B.update(0.43)
        D.update(0.3)
        munp.update(1.)
        muwp.update(0.63)
        #Best parameter setting: A=0.0863560529072 B=0.431991519569 D=0.326106143866
                        #munp=1.0 muwp=0.633067500674


    seed = {"A":A, "B":B, "D":D, "munp":munp, "muwp":muwp}
    
    # Print info in files (log, results)
    arguments_text = strArgs(args)
    parameters_text = strParameters(seed)
    logger.info("HillClimbing with {0} seeds and {1} iterations".format(args.seeds, args.iterations))
    logger.info(arguments_text)
    logger.info(parameters_text)
    resultsFile.write(arguments_text)
    resultsFile.write(parameters_text)
    resultsFile.write("#A\tB\tD\tmunp\tmuwp\t{0}_exp1\t{0}_exp2\t{0}_exp3\t{0}_sum\n".format(args.metric))
        
    

    
    # Random-restart (shotgun) hill climbing
    initial=-1.0 if args.metric == "pearsonr" else float(sys.maxint)
    best_eval_each={1:initial, 2:initial, 3:initial}
    best_eval_sum = sum(best_eval_each.values())
    best_weights = copy.deepcopy(seed)
    best_data = {}
    next_data = {}
    i = 0


    while i < args.seeds:
        i += 1
        vprint("Seed {0}".format(i))
        # initial seed
        if not args.forceSeed:
            for v in seed.values():
                r = randGlobal.myRandom.uniform(v.min, v.max)
                v.update(r)

        resultsFile.write("\n\n\n#New seed.\n")

        # stochastic hill climbing
        next_weights, next_eval_each, next_data = stochasticHillClimbing(copy.deepcopy(seed), args.iterations, args.jump,  experiments, candidates, args.metric, args.runsModel, args.nmax)

        # if next solution is better, keep it (along with its weights)
        next_eval_sum=sum(next_eval_each.values())
        if  better_than(next_eval_sum, best_eval_sum, args.metric):
            best_eval_each = next_eval_each
            best_eval_sum=next_eval_sum
            best_weights = copy.deepcopy(next_weights)
            best_data = copy.deepcopy(next_data)


    # Output
    summary(best_data, best_weights, args.metric, best_eval_each, experiments)
    #Plot
    #text_weights = "R&R: A={0:.3f} B={1:.3f} D={2:.3f}".format(best_weights["A"].value, best_weights["B"].value, best_weights["D"].value)
    #text_weights += "\n munp={0:.3f} muwp={1:.3f} ".format(best_weights["munp"].value, best_weights["muwp"].value)
#    for nExp, experiment in experiments.iteritems():
#        print "Plotting experiment ", nExp, "..."
#        experiment.plotPerformance(best_data[nExp], text_weights, "frankrnr_exp{0}_{1}.png".format(nExp,timestamp), best_eval_each[nExp], args.metric)
#        #if args.metric == "pearsonr":
#        #    experiment.scatterPlot(best_data, "scatter.png", "")
    resultsFile.close()
    
    print "Finished! Results in: ", os.getcwd()
    
    
    
if __name__ == "__main__":

    seed=time.time()
    randGlobal.initSeed(seed)

    vprint("Initiating hill climbing, for ALL experiments in Frank et al. 2010...")
    
    # Command line args
    parser = argparse.ArgumentParser()                                                                                                                          
    parser.add_argument("--seeds", type=int, required=False , nargs='?', default=2)
    parser.add_argument("--iterations", type=int, required=False , nargs='?', default=50)
    parser.add_argument("--jump", type=float, required=False , nargs='?', default=0.5)
    parser.add_argument("--metric", type=str,required=True, choices=['mse', 'pearsonr'])
    parser.add_argument("--forceSeed", action="store_true", default=False)
    parser.add_argument("--runsModel", type=int, required=False , nargs='?', default=10)
    parser.add_argument("--nmax", type=int, required=False , nargs='?', default=4)
    parser.add_argument("--candidatesFile", type=str, required=False, nargs="?", default="")
    args = parser.parse_args()

    main(args)
    

