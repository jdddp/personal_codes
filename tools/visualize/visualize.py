import os
import sys
import os.path as osp
import pdb
import glob


def get_url(abs_imgpath,port_num=8225):
    if not osp.isfile(abs_imgpath):
        print('{} is not a imgpath'.format(abs_imgpath))
    personal_url='http://192.168.1.126:%d'%port_num+'\\'
    if abs_imgpath.startswith('E:\\vsCodes\\'):
        return abs_imgpath.replace('E:\\vsCodes\\', personal_url)

def refrom_res(lst):
        """Format table of imagelist
        """
        # out = '<meta charset="utf-8">'
        # out='<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'
        out=''
        for i, (cs, prob) in enumerate(lst):
            img_url = get_url(cs)
            out += '<td><div><img src=\"%s\" width=200 height=200 border=1 alt=\"img\" /><br><a href=\"%s\"target=\"_blank\">%.3f</a></div></td>' \
                    % (img_url, img_url, prob)
        return out


def visulize_cslist(cs_list, filename, decend=1):
        """将cslist可视化
        Args:
            cs_list: 处理成dict of list, each item of is (imagepath, score) or just imagepath
            filename: html存储地址
            row*col: numbers of imgs in each html
        """
        output = open(filename, 'w')
        each_row, max_col = 10, 100 
        items, col_idx = [], 0
        
        if not (isinstance(cs_list[0], list) or isinstance(cs_list[0], tuple)):
            cs_list = [(cs, 1) for cs in cs_list]

        cs_list = sorted(cs_list, key=lambda tup:tup[1])[::-1] if decend == 1 else \
                 sorted(cs_list, key=lambda tup:tup[1])
                 
        for i, (cs, prob) in enumerate(cs_list):
            if len(items) < each_row:
                items.append([cs, prob])
                continue
            output.write("<table><tr><td>%d</td></tr>\n" % (col_idx))
            output.write('<tr>%s</tr>\n' % (refrom_res(items)))
            output.write('</table>\n')
            col_idx += 1
            items = []
            if col_idx > max_col:
                break
        
        #
        if len(items) > 0:
            output.write("<table><tr><td>%d</td></tr>\n" % (col_idx))
            output.write('<tr>%s</tr>\n' % (refrom_res(items)))
            output.write('</table>\n')
        output.close()
        return get_url(filename)

import pdb
def visualize_folder(imgdir, filename):
    lst = glob.glob(osp.join(imgdir, '*.jpg'))
    # pdb.set_trace()
    cslist = [(p, 1) for p in lst]
    visulize_cslist(cslist, filename)
    url = get_url(filename)
    print("[visualize_folder] %s,[path of html] %s" % (imgdir, url))
    return url

def visualize_folders(imgdir, htmlDir):
    if not osp.isdir(htmlDir):
        os.makedirs(htmlDir)
    

    foldername2html = {}
    for foldername in os.listdir(imgdir):
        folder = osp.join(imgdir, foldername)
        if osp.isdir(folder):
            imglst = glob.glob(osp.join(folder, '*.jpg'))
            cslist = [(p, 1) for p in imglst]
            filename = osp.join(htmlDir, "%04d_" % len(cslist) + foldername + '.html')
            visulize_cslist(cslist, filename, decend=1)
            foldername2html[foldername] = get_url(filename)
            print('[dirname] {}'.format(foldername),foldername2html[foldername])

    return foldername2html
if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong args')