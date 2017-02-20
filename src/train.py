import RnR
import Frank2010



class TrainAndEvaluate:

    def __init__(self,model):
        self.model = model

    def fit(self,cost_function,data,optimization_algorithm):
        pass






if __name__ == '__main__':
    expId = 1
    exp = Frank2010.experiment(expId)

    for cnd, expcnd in exp.cnd.iteritems():
        print Frank2010.CONDITION[expId] + ": " + str(cnd)
        print expcnd.stream



    rnr_model = RnR.RnRv2(A=0.04, B=0.3, C=0.3, D=0.3, nmax=4)
    for cnd, expcnd in exp.cnd.iteritems():
        candidates = rnr_model.memorizeOnline("##".join(expcnd.stimuli))
        print candidates

print(expcnd.test)