# python ./dcnn.py get_feature \
# /home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/Fluid_inclusions \
# /home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/clus_ans.txt \
# /home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/clus_ans.npy

python ./k_means.py cluster \
/home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/clus_ans.npy \
/home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/clus_ans.txt \
/home/linke/codes/localCodes/dataset/trainDatasetV1/crop_dir/clus_ans.json \
10
