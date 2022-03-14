## 查询
#fuser -v /dev/nvidia*

##kill
#kill -9 `ps -ef | grep train_face.py | awk '{print $2}'`

##kill_2
#ps -ef | grep gpu_burn | grep -v grep | awk '{print $2}' | xargs kill

##目录下文件夹数量统计
#find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done