import gies
import numpy as np
def GIES(X_list, B):
    interventions = []
    for i in range(B.shape[1]):
        interventions.append(list(np.where(B[:,i]==0)[0]))
    estimate,score = gies.fit_bic(X_list,interventions,phases = ['forward', 'backward', 'turning'])
    return estimate, None