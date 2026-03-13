import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing



def cuts(df):

    df = df[np.logical_and(df['x1']>=-3,df['x1']<=3)]
    df = df[np.logical_and(df['c']>=-0.3,df['c']<=0.3)]
    df = df[df['x1ERR']<=1.0]
    df = df[df['cERR']<=1.0]
    df = df[df['PKMJDERR']<=2.0]
    df = df[np.logical_and(df['FITPROB']>=0.05,df['FITPROB']<=1.0)]

    return df

for i in range(3):
    dir_ = 'SNANA_sims/TRAINING'+str(i)+'.FITRES'
    
    if i ==0:
        df  =cuts(pd.read_csv(dir_, comment="#", sep='\s+').sample(frac=1))[:3000000]
    else:
        df = pd.concat((df,cuts(pd.read_csv(dir_, comment="#", sep='\s+').sample(frac=1))[:3000000]))
    print(len(df))

    
dir_ = 'SNANA_sims/TRAINING_FLAT.FITRES'
df = pd.concat((df,cuts(pd.read_csv(dir_, comment="#", sep='\s+').sample(frac=1))[:1000000])).sample(frac=1)
print(len(df))

train_arr=np.zeros((len(df),0),dtype=np.float64)
#train_arr=np.zeros((len(df),0))
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
train_arr = np.append(train_arr,np.array(df['SIM_DLMAG']+df['SIM_MUSHIFT']-19.36500000433958).reshape(-1,1),axis=1)
train_arr = np.append(train_arr,np.array(-df['SIM_alpha']).reshape(-1,1),axis=1)
train_arr = np.append(train_arr,np.array(df['SIM_beta']).reshape(-1,1),axis=1)

np.save('SNANA_training_set.npy',train_arr)
