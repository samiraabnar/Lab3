class optimization:

    @staticmethod
    def greedy_search(self,model,data
                      #you may add extra parameters if you need
                     ):
        raise NotImplementedError

    @staticmethod
    def hill_climbing_1d():
        raise NotImplementedError


    @staticmethod
    def hill_climbing_nd(maxIterations, maxJump, experiment, candidates, metric, runsModel, nmax):
        raise NotImplementedError


    @staticmethod
    def get_next_weight():
        var = min(v.max - best_wT[k].value, best_wT[k].value - v.min) * maxJump
        next_wT[k].update(randGlobal.myRandom.gauss(best_wT[k].value, var))