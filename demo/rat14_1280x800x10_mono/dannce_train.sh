# AVG
cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono
python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_1280x800x10_config.yaml --gpu-id 2

# MAX
cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono
python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_1280x800x10_max_config.yaml --gpu-id 0,1,2,3
