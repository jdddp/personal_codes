import os,sys
import os.path as osp
import json
import pandas as pd

def excel2json(inexcel, jsonPath):
    '''convert excel to json
    '''
    df_all=pd.read_excel(inexcel, sheet_name=None)
    ansDct={}
    for sheet_name, df in df_all.items():
        dct_lst=df.to_dict('records')
        ansDct[sheet_name]=dct_lst
        print('sheet_name {}, get {} records'.format(sheet_name, len(dct_lst)))
    
    with open(jsonPath,'w',encoding='utf-8')as f:
        f.write(json.dumps(ansDct))

def json2excel(jsonPath, outexcel):
    '''convert json to excel
    '''
    res=json.loads(open(jsonPath).read())
    with pd.ExcelWriter(outexcel)as writer:
        for name, sgDct in res.items():
            df=pd.DataFrame.from_dict(sgDct)
            df.to_excel(writer, sheet_name=name)

if __name__ == '__main__':
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')