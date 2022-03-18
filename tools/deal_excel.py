#using coding=utf-8
#python3
import os
import sys
import json
import pdb
import glob
import collections
import pandas as pd
import os.path as osp
import copy

def excel2json(inexcel, outjson):
    """
    convert excel to json
    """
    df_all = pd.read_excel(inexcel, sheet_name=None)
    class_infos = {}
    #sheetname: labelmap, class31, class20
    for sheet_name, df in df_all.items():
        items = df.to_dict('records')
        # new_items = trans_dict(items)
        class_infos[sheet_name] = items
        print("sheet_name {}, get {} records".format(sheet_name, len(class_infos[sheet_name])))

    json.dump(class_infos, open(outjson, 'w'), indent=4)
    return class_infos


def merge_dict2excel(inexcel, indict, inkeyname, junctionkey, outexcel):
    """
    convert excel to json
    """
    df_all = pd.read_excel(inexcel, sheet_name=None)
    class_infos = {}
    #sheetname: labelmap, class31, class20
    for sheet_name, df in df_all.items():
        items = df.to_dict('records')
        newitems = []
        for item in items:
            label = item[junctionkey]
            newvalue = indict.get(label, '')
            item.update({inkeyname:newvalue})
            newitems.append(item)
        class_infos[sheet_name] = newitems
        # pdb.set_trace()
        print("sheet_name {}, get {} records".format(sheet_name, len(class_infos[sheet_name])))

    
    with pd.ExcelWriter(outexcel) as writer:  
        for schema_name, class_info in class_infos.items():
            df = pd.DataFrame.from_dict(class_info)
            df.to_excel(writer, sheet_name=schema_name, float_format="%.2f")
    return class_infos

def json2excel(injson, outexcel):
    """
    convert json to excel
    """
    class_infos = json.load(open(injson, 'r'))
    if isinstance(class_infos, list):
        class_infos = {"main":class_infos}
    
    with pd.ExcelWriter(outexcel) as writer:  
        for schema_name, class_info in class_infos.items():
            df = pd.DataFrame.from_dict(class_info)
            df.to_excel(writer, sheet_name=schema_name, float_format="%.2f")


def readexcel2txt(inexcel, outdir, query="三级标签", search_words="检索词"):
    """
    convert excel to json
    """
    df_all = pd.read_excel(inexcel, sheet_name=None)
    class_infos = {}
    #sheetname: labelmap, class31, class20
    for sheet_name, df in df_all.items():
        items = df.to_dict('records')
        for item in items:
            label = item[query]
            query = item[search_words]
            with open(osp.join(outdir, label+'.txt'), 'w') as f:
                ss = query.split('、')
                print('{} get {}'.format(label, len(ss)))
                f.write('\n'.join(ss))
        
def gen_wangyidict(inexcel, outjson):
    df_all = pd.read_excel(inexcel, sheet_name=None)
    class_infos = {}
    #sheetname: labelmap, class31, class20
    for sheet_name, df in df_all.items():
        items = df.to_dict('records')
        # new_items = trans_dict(items)
        class_infos[sheet_name] = items
        print("sheet_name {}, get {} records".format(sheet_name, len(class_infos[sheet_name])))

    labelmap ={}
    for item in class_infos['label']:
        labelmap[item["label"]] = item["classname"]
    for item in class_infos['sublabel']:
        # label = item["label"]
        labelmap[item["sublabel"]] = item["classname"]
    json.dump(labelmap, open(outjson, 'w'), indent=4)




if __name__ == '__main__':
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr,'update_diku.py addfolder/addimg'


# python parse_excel.py  excel2json  output/20211117_stats.xlsx output/stats.json
# python parse_excel.py  json2excel  ooutput/stats.json output/20211117_stats.xlsx
