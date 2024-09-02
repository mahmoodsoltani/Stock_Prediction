

import pandas as pd
import numpy as np
#### Parameters
InputFile ='E:/Project/AlphaNiroo/Dataset/test_down_.xlsx' 
OutputFile ='E:/Project/AlphaNiroo/Dataset/testresult.xlsx'
ExtemrsCount=5
################################

def getdata(path):
    excel_test_data =  pd.read_excel(path)
    return pd.DataFrame(excel_test_data, columns=['CHART SYMBOL','OPEN','HIGH','LOW','CLOSE','Time','i','MAX EXTREMUM','MIN EXTREMUM'])
def Is_PullBack_DOWN(val,index,Check_p5,percent):
    extrema = np.array(val.split(),dtype=float)
    extrema_index = np.array(index.split(),dtype=np.int32)
    e6= extrema[-1]
    #e6_index = extrema_index[-1]
    for i in range(len(extrema)-2,0,-1):
        e5 = extrema[i]
        e5_index = extrema_index[i]
        if e5<=e6: 
            return '0'
        flag= False
        for z in range(len(extrema)-1,i,-1):
            if e5<=extrema[z] or e6>extrema[z]:
                flag=True
                break
        if flag:
            continue
        for j in range(i-1,0,-1):
            e4 = extrema[j]
            e4_index = extrema_index[j]
            if e5<=e4  : 
                continue
            flag= False
            for z in range(i-1,j,-1):
                if e5<=extrema[z] or e4>=extrema[z]:
                    flag=True
                    break
            if flag:
                continue
            for k in range(j-1,0,-1):
                e3 = extrema[k]
                e3_index = extrema_index[k]
                if e3<=e4 or e3<=e5 or e3<=e6 or e6>=e3-((e3-e4)*percent/100) or e5<=e3-((e3-e4)*percent/100):
                    continue
                flag= False
                for z in range(j-1,k,-1):
                    if e4<=extrema[z] or e3>=extrema[z]:
                        flag=True
                        break
                if flag:
                    continue
                for l in range(k-1,0,-1):
                    e2 = extrema[l]
                    e2_index = extrema_index[l]
                    if Check_p5:
                        if e2>=e5:
                          continue
                    flag= False
                    for z in range(k-1,l,-1):
                        if e3<=extrema[z] or e2>=extrema[z]:
                            flag=True
                            break
                    if flag:
                        continue
                    if e2>=e3 or e2<=e4  :
                        continue
                    for m in range(l-1,-1,-1):
                        e1 = extrema[m]
                        e1_index = extrema_index[m]
                        flag= False
                        for z in range(k-1,m,-1):
                            if e1<=extrema[z]:
                                flag=True
                                break
                        if flag:
                            continue
                        flag= False
                        for z in range(l-1,m,-1):
                            if e1<=extrema[z] or e2>=extrema[z] :
                                flag=True
                                break
                        if flag:
                            continue
                        if e1<=e2 or e1<=e3:
                            continue
                        else:
                            # return str(e6_index)+' '+str(e5_index)+' '+str(e4_index)+' '+str(e3_index)+' '+str(e2_index)+' '+str(e1_index)
                            return str(int(round((e3-e6)/(e3-e4),2)*100))+ ',' +str(e5_index)+' '+str(e4_index)+' '+str(e3_index)+' '+str(e2_index)+' '+str(e1_index)
    return str(0)
def Is_PullBack_UP(val,index,Check_p5,percent):
    extrema = np.array(val.split(),dtype=float)
    extrema_index = np.array(index.split(),dtype=np.int32)
    e6= extrema[-1]
    #e6_index = extrema_index[-1]
    for i in range(len(extrema)-2,0,-1):
        e5 = extrema[i]
        e5_index = extrema_index[i]
        if e5>=e6: 
            return '0'
        flag= False
        for z in range(len(extrema)-1,i,-1):
            if e5>=extrema[z] or e6<extrema[z]:
                flag=True
                break
        if flag:
            continue
        for j in range(i-1,0,-1):
            e4 = extrema[j]
            e4_index = extrema_index[j]
            if e5>=e4 : 
                continue
            flag= False
            for z in range(i-1,j,-1):
                if e5>=extrema[z] or e4<=extrema[z]:
                    flag=True
                    break
            if flag:
                continue

            for k in range(j-1,0,-1):
                e3 = extrema[k]
                e3_index = extrema_index[k]
                if e3>=e4 or e3>=e5 or e3>=e6 or e6<=((e4-e3)*percent/100)+e3 or e5>=((e4-e3)*percent/100)+e3:
                    continue
                flag= False
                for z in range(j-1,k,-1):
                    if e3>=extrema[z] or e4<=extrema[z]:
                        flag=True
                        break
                if flag:
                    continue
                for l in range(k-1,0,-1):
                    e2 = extrema[l]
                    e2_index = extrema_index[l]
                    if Check_p5:
                        if e2<=e5:
                          continue
                    flag= False
                    for z in range(k-1,l,-1):
                        if e3>=extrema[z] or e2<=extrema[z]:
                            flag=True
                            break
                    if flag:
                        continue
                    if e2<=e3 or e2>=e4  :
                        continue
                    for m in range(l-1,-1,-1):
                        e1 = extrema[m]
                        e1_index = extrema_index[m]
                        flag= False
                        for z in range(k-1,m,-1):
                            if e1>=extrema[z]:
                                flag=True
                                break
                        if flag:
                            continue
                        flag= False
                        for z in range(l-1,m,-1):
                            if e3>=extrema[z] or e2<=extrema[z]:
                                flag=True
                                break
                        if flag:
                            continue
                        if e1>=e2 or e1>=e3:
                            continue
                        else:
                            # return str(e6_index)+' '+str(e5_index)+' '+str(e4_index)+' '+str(e3_index)+' '+str(e2_index)+' '+str(e1_index)
                            return str(int(round((e6-e3)/(e4-e3),2)*100))+ ',' + str(e5_index)+' '+str(e4_index)+' '+str(e3_index)+' '+str(e2_index)+' '+str(e1_index)
    return str(0)
def find_pullback(window,Path,Check_p5,percent):
    pull_up=0
    Pull_down=0
    # data= getdata('D:\Project\Stocks\Dataset/Test_Exterms.xlsx')
    data= getdata(Path)
    Max = data.loc[(data['MAX EXTREMUM'])!=0].index.tolist()
    Min =data.loc[(data['MIN EXTREMUM'])!=0].index.tolist()
    Exterma_Value = pd.concat([data.loc[Max, 'HIGH'], data.loc[Min, 'LOW']]).sort_index()
    # data.iloc[:,2].plot(color='green')
    # data.iloc[:,3].plot(color='green')
    # plt.scatter(Exterma_Value.index,Exterma_Value.values,color='gray')
    PullBack_UP = ["" for i in range(len(data))]
    PullBack_Down = ["" for i in range(len(data))]
    for i in range(window,len(Exterma_Value)+1):
        _strVal=''
        _strindex=''
        sub_Exterma = Exterma_Value.iloc[i-window:i] 
        for k in range (0,len(sub_Exterma )):
            _strVal = _strVal + ' '+ str(sub_Exterma.iloc[k])
            _strindex = _strindex + ' '+ str(Exterma_Value.index[i-window+k])
        d=0
        if  k+1 == len(Exterma_Value):
            d = len(data)
        else:
            d = Exterma_Value.index[i]
        for zz in range (sub_Exterma.index[k]+1 , d):
            k = Is_PullBack_UP(_strVal+ ' ' +str(data.loc[zz,'CLOSE']),_strindex+ ' '+ str(zz),Check_p5,percent)  
            if k != '0':
                pull_up = pull_up+1
                print(k + 'UP')
                extrema_index = np.array(k.split(',')[1].split(),dtype=np.int32)
                PullBack_UP[zz] =PullBack_UP[zz]+'P_6_'+str(pull_up)+','
                for _i in range (0,len(extrema_index)):
                    PullBack_UP[extrema_index[_i]]=PullBack_UP[extrema_index[_i]]+'P_'+str(5-_i)+'_'+str(pull_up)+','
                break
            k = Is_PullBack_DOWN(_strVal+ ' ' +str(data.loc[zz,'CLOSE']),_strindex+ ' '+ str(zz),Check_p5,percent)  
            if k != '0':
                Pull_down = Pull_down+1
                print(k+ 'Down')
                extrema_index = np.array(k.split(',')[1].split(),dtype=np.int32)
                PullBack_Down[zz] =PullBack_Down[zz]+'P_6_'+str(Pull_down)+','
                for _i in range (0,len(extrema_index)):
                    PullBack_Down[extrema_index[_i]]=PullBack_Down[extrema_index[_i]]+'P_'+str(5-_i)+'_'+str(Pull_down)+','
                break
    data['PullBack_UP'] = PullBack_UP
    data['PullBack_DOWN'] = PullBack_Down
    data.to_excel(OutputFile)
find_pullback(ExtemrsCount,InputFile,0,75)
