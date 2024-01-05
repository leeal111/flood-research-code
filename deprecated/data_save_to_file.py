import os
import pandas as pd

dataAnanlyzePath = os.path.join('temp', 'dataAnanlyze')
windowSavePath=os.path.normpath(r'C:\Users\LeeaL\Desktop')

def toTempOutputTxt(x):
    with open(os.path.join(dataAnanlyzePath,'output.txt')) as file:
        for item in x:
            file.write(str(item)+'\n')

def toExcel(data,file,ifT=False):
    df = pd.DataFrame(data)
    if ifT==True:
        df=df.T
    df.to_excel(file, index=False)

def toTempExcelOnWindow(data,ifT=False):
    basePath=os.path.join(windowSavePath,'temp.xlsx')
    df = pd.DataFrame(data)
    if ifT==True:
        df=df.T
    df.to_excel(basePath, index=False)

def toTempExcel(data,ifT=False):
    basePath=os.path.join(dataAnanlyzePath,'temp.xlsx')
    df = pd.DataFrame(data)
    if ifT==True:
        df=df.T
    df.to_excel(basePath, index=False)

if __name__=="__main__":
    df=pd.DataFrame([[42] * 5] * 3)
    print(df)