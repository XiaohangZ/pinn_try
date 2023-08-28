def checkDirMake(directory):
    #print(directory)
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def unnorm(tensor,mean,std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    return tensor