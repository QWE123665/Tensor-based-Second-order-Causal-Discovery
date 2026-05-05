from causallearn.search.ScoreBased.GES import ges

def GES(X):
    cg = ges(X) 
    return cg["G"].graph, None