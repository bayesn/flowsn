import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing

import copy

# import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing


tot = 0
max_ =0
min_ = 1000
lens = []

best =[]# import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing


tot = 0
max_ =0
min_ = 1000
lens = []

best =[]
def cuts(df):

    df = df[np.logical_and(df['x1']>=-3,df['x1']<=3)]
    df = df[np.logical_and(df['c']>=-0.3,df['c']<=0.3)]
    df = df[df['x1ERR']<1.0]
    df = df[df['cERR']<1.0]
    df = df[df['PKMJDERR']<2.0]
    df = df[np.logical_and(df['FITPROB']>=0.05,df['FITPROB']<=1.0)]

    return df


for i in range(1):
    
    add_str = '0'+str(i+1) if i<9 else str(i+1)
    add_str = '0'+add_str if i<99 else add_str
    dir_ = 'output/PIP_SBI_PLOT_PLASTICC_SIMDATA-0'+add_str+'/FITOPT000.FITRES'
    
    df  = cuts(pd.read_csv(dir_, comment="#", sep='\s+').sample(frac=1))

    train_arr=np.empty((len(df),0),dtype=np.float64)
    train_arr = np.append(train_arr,np.array(df['zHEL']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['zHD']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['zHDERR']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['mB']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['c']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['x1']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['mBERR']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['cERR']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['x1ERR']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(-2.5/np.log(10)*df['COV_c_x0']/df['x0']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(-2.5/np.log(10)*df['COV_x1_x0']/df['x0']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['COV_x1_c']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['SIM_DLMAG']-19.36500000433958).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(-df['SIM_alpha']).reshape(-1,1),axis=1)
    train_arr = np.append(train_arr,np.array(df['SIM_beta']).reshape(-1,1),axis=1)

    if i == 0:
        new_train_arr = copy.deepcopy(train_arr)
    else:
        new_train_arr = np.append(new_train_arr,train_arr,axis=0)
    print(i,' ',len(new_train_arr))

np.save('test_single.npy',new_train_arr)
    

