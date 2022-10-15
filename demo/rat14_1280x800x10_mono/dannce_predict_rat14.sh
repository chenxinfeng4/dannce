# cd demo/rat14_1280x800x10_mono_shank3

python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict ../../configs/dannce_rat14_1280x800x10_config.yaml


vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_19-31-21_SHANK20_HetxHet
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_18-59-15_SHANK20_KOxKO # runned
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_17-51-06_SHANK21_KOxKO # running
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_18-25-10_SHANK21_HetxHet 
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/half_processed/2022-02-26_17-51-06_SHANK21_KOxKO

vfile=/home/liying_lab/chenxinfeng/DATA/tao_rat/data/20220613-side6-addition/TPH2-KO-multiview-202201/male/cxf_batch/bwt-wwt-01-18_12-54-05

python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x10_max_config.yaml --video-file $vfile.mp4 --gpu-id 3

python -m lilab.dannce.s4_videopredictpkl2matcalibpkl ${vfile}_dannce_predict.pkl
python -m lilab.mmpose.s3_matcalibpkl_2_video2d ${vfile}.matcalibpkl --iview 1
python -m lilab.smoothnet.s1_matcalibpkl2smooth ${vfile}.matcalibpkl

# ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-2*.mp4 \
ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/tmp_video/*.segpkl \
| sed 's/segpkl/mp4/g' | cat -n   \
| xargs -P 4 -l bash -c 'echo python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x10_max_config.yaml --video-file $1 --gpu-id $(($0%4))'
# | xargs -l bash -c 'echo python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x10_max_config.yaml --video-file $1 --gpu-id $(($0%4))' \


# 1. prediction. copy multi
# Set the 'choosecuda order'. And copy paste to each pannel.
ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25*.mp4 \
| grep -v mask | grep -v sktdraw | cat -n   \
| xargs -P 4 -l bash -c 'echo python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x10_max_config.yaml --video-file $1 --gpu-id 0'

vfilesHost=`ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022*.mp4 \
| grep -v mask | grep -v sktdraw | grep -v com3d`
vfilesFini=`ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/*.mp4 \
| grep _1_sktdraw.mp4 | sed 's/_1_sktdraw//'`
vfiles=`comm -23 <(echo "$vfilesHost") <(echo "$vfilesFini")`

echo "$vfiles" | cat -n   \
| xargs -P 0 -l bash -c 'echo python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x10_max_config.yaml  --gpu-id 0 --video-file $1'


# 2. Convert file, one hit
ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25*.mp4 \
| grep -v mask | grep -v sktdraw | sed 's/.mp4/_dannce_predict.pkl/' \
| xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl

echo "$vfiles" | sed 's/.mp4/_dannce_predict.pkl/' \
| xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl


# 3. Video generate. copy multi
ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25*.mp4 \
| grep -v mask | grep -v sktdraw | sed 's/.mp4/.matcalibpkl/' \
| xargs -P 0 -l -r bash -c 'echo python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1'

echo "$vfiles" | sed 's/.mp4/.matcalibpkl/' \
| xargs -P 0 -l -r bash -c 'echo python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1'




# 4. Smooth, one hit
ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25*.mp4 \
| grep -v mask | grep -v sktdraw | sed 's/.mp4/.matcalibpkl/' \
| xargs -l -r python -m lilab.smoothnet.s1_matcalibpkl2smooth

vfilesHost=`ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/*.matcalibpkl | grep -v smoothed`
vfilesFini=`ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/*.smoothed.matcalibpkl | sed 's/.smoothed//'`
vfiles=`comm -23 <(echo "$vfilesHost") <(echo "$vfilesFini")`
echo "$vfiles" | xargs -l -r python -m lilab.smoothnet.s1_matcalibpkl2smooth
echo "$vfiles" | sed 's/.mp4/.matcalibpkl/' | xargs -l -P 4 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth


# 5 Concat video
echo "$vfiles" | sed 's/.matcalibpkl//' | xargs -l -r bash -c 'echo python -m lilab.cvutils.concat_videopro $0_1_sktdraw.mp4 $0_1_sktdraw_smoothed.mp4'