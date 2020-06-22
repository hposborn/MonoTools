import numpy as np
import os
import re
import socket
import pandas as pd

df=pd.read_csv("~/MonoTools/data/tables/2020_monos_unq.csv")

allTess = df.loc[df['mission'].str.lower()=='tess']
nonTESS = df.loc[(df['mission'].str.lower()=='k2')|(df['mission'].str.lower()=='kepler')]

if 'antares' in socket.gethostname():
    subset=nonTESS.iloc[1::2]
    runfileloc='/export/home/hosborn/data/run/'
    n_runs=3
elif 'sirius' in socket.gethostname():
    subset=nonTESS.iloc[0::2]
    runfileloc='/nfs/ger/home/hosborn/MonoTools_run/'
    n_runs=4
elif 'pdo1' in socket.gethostname():
    subset=allTess.iloc[0::2]
    runfileloc='/pdo/users/hosborn/MonoTools_run/'
    n_runs=3
elif 'pdo6' in socket.gethostname():
    subset=allTess.iloc[1::2]
    runfileloc='/pdo/users/hosborn/MonoTools_run/'
    n_runs=4
    
non_decimal = re.compile(r'[^\d.]+')

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}

df=df.loc[df['mission']!='CoRoT']
os.system("rm "+runfileloc+"/*.sh")

n=0
for id,row in subset.iterrows():
    icid=str(int(float(non_decimal.sub('',row['id']))))
    if 'pdo6' in socket.gethostname():
        runfile=runfileloc+id_dic[row['mission']]+icid.zfill(11)+"_pdo1_run.sh"
        runallfile=runfileloc+"runall_"+str(int(np.ceil(n_runs*n/len(subset))))+"_pdo1.sh"
    elif 'pdo1' in socket.gethostname():
        runfile=runfileloc+id_dic[row['mission']]+icid.zfill(11)+"_pdo6_run.sh"
        runallfile=runfileloc+"runall_"+str(int(np.ceil(n_runs*n/len(subset))))+"_pdo6.sh"
    else:
        runfile=runfileloc+id_dic[row['mission']]+icid.zfill(11)+"_run.sh"
        runallfile=runfileloc+"runall_"+str(int(np.ceil(n_runs*n/len(subset))))+".sh"
    with open(runfile,"w") as fo:
        if 'pdo' in socket.gethostname():
            fo.write("#!/bin/sh\nsource ~.bashrc\ntid_get_mono "+icid+"\nsource ~/anaconda3/etc/profile.d/conda.sh\nconda activate monoenv\ncd ~/MonoTools\npython main.py "+icid+" "+row['mission'].lower()+"\n")
        else:
            fo.write("#!/bin/sh\nsource ~/anaconda3/etc/profile.d/conda.sh\nconda activate monoenv\ncd ~/MonoTools\npython main.py "+icid+" "+row['mission'].lower()+"\n")
    if not os.path.exists(runallfile):
        with open(runallfile,"w") as fo2:
            fo2.write("#!/bin/sh\n")
    with open(runallfile,"a") as fo2:
        fo2.write("bash "+runfile+"\n")
    n+=1

os.system("chmod +x "+runfileloc+"*.sh")