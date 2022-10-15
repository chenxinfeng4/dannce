cd /home/liying_lab/chenxinfeng/DATA/dannce/demo_single/rat14_1280x800x4_mono
# ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-2*.mp4 \
vfiles=`/mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/0921/*.segpkl`
vfiles=/mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/0921/male.segpkl

echo "$vfiles" | sed 's/.segpkl/.mp4/' | cat -n  \
| xargs -P 8 -l bash -c 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x4_max_config.yaml --video-file $1 --gpu-id $(($0%4))'


# 2. Convert file, one hit
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' \
| xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl


# 3. Video generate. copy multi
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' \
| xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 '

# 4A. Smooth, one hit
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 4 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_shorter
echo "$vfiles" | sed 's/.segpkl/.smoothed_w16.matcalibpkl/' | xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_w16 --maxlen 9000'

# 4B. Smooth, one hit 
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 4 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth

# 5 Concat video
echo "$vfiles" | sed 's/.segpkl//' | xargs -l -r bash -c 'python -m lilab.cvutils.concat_videopro $0_1_sktdraw.mp4 $0_1_sktdraw_smoothed.mp4'
