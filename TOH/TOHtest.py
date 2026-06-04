import numpy as np

Tower = np.array(np.array((np.flip(np.arange(1, 4))))),np.array([]),np.array([])

obs = Tower
for z in range(3):
    if(len(obs[z])!=3):
        for i in range(3-len(obs[z])):
            np.append(obs[z], 0)
print(obs)
#np.insert(Tower[1],Tower[0][-1])
#np.delete(Tower[0],-1)

#np.append()