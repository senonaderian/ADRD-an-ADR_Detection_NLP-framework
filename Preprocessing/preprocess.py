import numpy as np
import pandas as pd
df = pd.read_csv(r'ADR.csv')
a,b = df.shape
df.columns
df = df.drop(['id','sentence_index','sentences'],axis=1)
df.columns
review = 1
d={}

while review <= a:
    #print(review)
    if 'lexapro' in df['drug_id'][review]:
        for i in range(1,220):
            r = 'lexapro.'+str(i)
            #print(r)
            c = df.loc[df['drug_id']== r]
            array1 = c.values  
            d[r] = array1[:,1:]
        review = 589
        
        

                
    elif 'zoloft' in df['drug_id'][review]:
        for i in range(1,214):
            r = 'zoloft.'+str(i)
            #print(r)
            c = df.loc[df['drug_id']== r]
            array1 = c.values  
            d[r] = array1[:,1:]
        review = 1060


    elif 'cymbalta' in df['drug_id'][review]:
        for z in range(1,232):
            r = 'cymbalta.'+str(z)
            #print(r)
            c = df.loc[df['drug_id']== r]
            array1 = c.values  
            d[r] = array1[:,1:]

        review = 1676


    elif 'EffexorXR' in df['drug_id'][review]:
        for k in range(1,229):
            r = 'EffexorXR.'+str(k)
            #print(r)
            c = df.loc[df['drug_id']== r]
            array1 = c.values  
            d[r] = array1[:,1:]
        review = a+1
#print(d)
print(df.columns)
for r in df['drug_id'][:]:
    #print(r)
    z = d[r]
    #print(z)
    x = []
    for i in z:
        #print(i)
        if i.any() != '.':
            for j in i:
                j.lower()
            x.append(i)
    print(r)
    print(len(np.unique(x)))
