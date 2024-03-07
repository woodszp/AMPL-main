# tiered
python -m torch.distributed.launch --nproc_per_node=8 main_ampl.py --data_path /home/xxx/Data/tiered_imagenet/train --pretrained_path /home/xxx/Main/AMPL_Result/pretrain --pretrained_file checkpoint0150.pth --output_dir /home/xxx/Main/AMPL_Result/TIERED480_phase2 --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 64 --pred_start_epoch 0 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4

# prototype
python eval_ampl.py --server mini --num_shots 5 --ckp_path /root/autodl-nas/FSVIT_results/MINI480_phase2 --ckpt_filename checkpoint0040.pth --output_dir /root/autodl-nas/FSVIT_results/MINI480_prototype --evaluation_method cosine --iter_num 10000

# classifier
python eval_ampl.py --server mini --num_shots 5 --ckp_path /home/xxx/Results/AMPL/pretrain/mini/ --ckpt_filename checkpoint0040.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000





#######
### SET GPU RUN 
CUDA_VISIBLE_DEVICES=0,1

# self-surpervised
# mini
python -m torch.distributed.launch --nproc_per_node=8 main_ampl.py --data_path /root/autodl-tmp/mini-imagenet-480/train --output_dir /root/autodl-nas/FSVIT_results/MINI480 --evaluate_freq 50 --visualization_freq 50 --use_fp16 True --lr 0.0005 --epochs 1600 --lambda1 1 --lambda2 0 --lambda3 1 --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4

# tiered
python -m torch.distributed.launch --nproc_per_node=8 main_ampl.py --data_path /root/autodl-tmp/tiered_imagenet-480/train --output_dir /root/autodl-nas/FSVIT_results/TIERED480 --evaluate_freq 50 --visualization_freq 50 --use_fp16 True --lr 0.0005 --epochs 1600 --lambda1 1 --lambda2 0 --lambda3 1 --pred_shape rand --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4

# CIFARFS
python -m torch.distributed.launch --nproc_per_node=8 main_ampl.py --data_path /root/autodl-tmp/cifar-fs-84/train --output_dir /root/autodl-nas/FSVIT_results/CIFARFS84 --evaluate_freq 50 --visualization_freq 50 --use_fp16 True --lr 0.0005 --epochs 1600 --lambda1 1 --lambda2 0 --lambda3 1 --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256

# FC100
python -m torch.distributed.launch --nproc_per_node=8 main_ampl.py --data_path /root/autodl-tmp/FC100-84/train --output_dir /root/autodl-nas/FSVIT_results/FC10084 --evaluate_freq 50 --visualization_freq 50 --use_fp16 True --lr 0.0005 --epochs 1600 --lambda1 1 --lambda2 0 --lambda3 1 --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256


## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ##
# surpervised
# For replay
# mini
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/miniImageNet/train --pretrained_path /home/xxx/Results/AMPL/pretrain/mini/ --pretrained_file checkpoint0040.pth --output_dir /home/xxx/Results/AMPL/MINI480_phase2 --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --partition val --saveckp_freq 5

# tiered
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/tiered_imagenet/train/ --pretrained_path /home/xxx/Results/AMPL/pretrain/tiered/ --pretrained_file checkpoint0150.pth --output_dir /home/xxx/Results/AMPL/TIERED480_phase2 --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --start_epoch 0 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4


# CIFARFS
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /root/autodl-tmp/cifar-fs-84/train --pretrained_path /root/autodl-nas/FSVIT_results/CIFARFS84 --pretrained_file checkpoint0800.pth --output_dir /home/xxx/Results/AMPL/CIFARFS84_phase2 --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.5 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256


# FC100
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /root/autodl-tmp/FC100-84/train --pretrained_path /root/autodl-nas/FSVIT_results/FC10084 --pretrained_file checkpoint0900.pth --output_dir /home/xxx/Results/AMPL/FC10084_phase2 --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256



## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ##
# surpervised
# add our model 
# mini
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/miniImageNet/train --pretrained_path /home/xxx/Results/AMPL/pretrain/mini --pretrained_file checkpoint0040.pth --output_dir /home/xxx/Results/AMPL/MINI480_AMPL --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 45 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --partition val --saveckp_freq 5

# tiered
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/tiered_imagenet/train --pretrained_path /home/xxx/Results/AMPL/pretrain/tiered --pretrained_file checkpoint0150.pth --output_dir /home/xxx/Results/AMPL/TIERED480_phase2 --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --pred_start_epoch 0 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4


# CIFARFS
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/cifarfs/train --pretrained_path /home/xxx/Results/AMPL/pretrain/CIFARFS --pretrained_file checkpoint0040.pth --output_dir /home/xxx/Results/AMPL/CIFARFS84_phase2 --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.5 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 45 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/cifarfs/train --pretrained_path /home/xxx/Results/AMPL/pretrain/CIFARFS --pretrained_file checkpoint0040.pth --output_dir /home/xxx/Results/AMPL/CIFARFS84_source --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.5 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256

# FC100
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/FC100/train --pretrained_path /home/xxx/Results/AMPL/pretrain/FC100 --pretrained_file checkpoint0060.pth --output_dir /home/xxx/Results/AMPL/FC10084_phase2 --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 45 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main_ampl.py --data_path /home/xxx/Data/FC100/train --pretrained_path /home/xxx/Results/AMPL/pretrain/FC100 --pretrained_file checkpoint0060.pth --output_dir /home/xxx/Results/AMPL/FC10084_source --evaluate_freq 5 --visualization_freq 5 --saveckp_freq 5 --use_fp16 True --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --lambda3 0 --supervised_contrastive --batch_size_per_gpu 85 --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --image_size 256 256



## test
# /home/xxx/Results/AMPL/MINI480_phase2_n1_2023-11-09_10-56-10/
# test basline for pretrain mini 5-shot 88.25% +- 0.09%
# test basline for training mini 5-shot 86.41% +- 0.09%
# test AMPL for trainng mini 5-shot 88.47% +- 0.09%

python eval_ampl.py --server mini --num_shots 5 --ckp_path /home/xxx/Results/AMPL/MINI480_phase2_n1_2023-11-09_10-56-10/ --ckpt_filename checkpoint0145.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000
 # AMPL 145.pth 85.62% +- 0.10%
python eval_ampl.py --server mini --num_shots 5 --ckp_path /home/xxx/Results/AMPL/MINI480_AMPL_n1_2023-11-10_00-02-39/ --ckpt_filename checkpoint0145.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000
python eval_ampl.py --server mini --num_shots 5 --ckp_path /home/xxx/Results/AMPL/MINI480_AMPL_n1_2023-11-10_00-02-39/ --ckpt_filename checkpoint0005.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000


python eval_ampl.py --server mini --num_shots 1 --ckp_path /home/xxx/Results/AMPL/MINI480_AMPL_n1_2023-11-10_00-02-39/ --ckpt_filename checkpoint0145.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000
# mini 10000 Test Acc at 100= 73.36% +- 0.18%
python eval_ampl.py --server mini --num_shots 1 --ckp_path /home/xxx/Results/AMPL/MINI480_AMPL_n1_2023-11-10_00-02-39/ --ckpt_filename checkpoint0005.pth --output_dir /home/xxx/Results/AMPL/MINI480_classifier --evaluation_method classifier --iter_num 1000

# test basline for training tiered 5-shot 86.41% +- 0.09%
# test AMPL for trainng tiered 5-shot 

# tiered 10000 Test Acc at 100= 67.67% +- 0.20%
python eval_ampl.py --server tiered --num_shots 1 --ckp_path /home/xxx/Results/AMPL/pretrain/tiered --ckpt_filename checkpoint0150.pth --output_dir /home/xxx/Results/AMPL/TIERED480_classifier --evaluation_method classifier --iter_num 1000


# mini AMPL 005 10000 Test Acc at 100= 66.24% +- 0.17% 

# mini 


# 
python eval_ampl.py --server fc100 --num_shots 5 --ckp_path /home/xxx/Results/AMPL/FC10084_phase2_n1_2023-11-11_20-09-08 --ckpt_filename checkpoint0145.pth --output_dir /home/xxx/Results/AMPL/FC100_classifier --evaluation_method classifier --iter_num 1000
# 10000 Test Acc at 100= 67.79% +- 0.16%
python eval_ampl.py --server fc100 --num_shots 5 --ckp_path /home/xxx/Results/AMPL/FC10084_phase2_n1_2023-11-11_20-09-08 --ckpt_filename checkpoint0005.pth --output_dir /home/xxx/Results/AMPL/FC100_classifier05 --evaluation_method classifier --iter_num 1000

# CIFAR-FS
# Baseline 10000 Test Acc at 100= 69.63% +- 0.18%
# Baseline 10000 Test Acc at 100= 87.10% +- 0.13%
python eval_ampl.py --server fs --num_shots 1 --ckp_path /home/xxx/Results/AMPL/CIFARFS84_source_n1_2023-11-14_12-36-57 --ckpt_filename checkpoint0139.pth --output_dir /home/xxx/Results/AMPL/FS_classifier_1shot --evaluation_method classifier --iter_num 1000
python eval_ampl.py --server fs --num_shots 5 --ckp_path /home/xxx/Results/AMPL/CIFARFS84_source_n1_2023-11-14_12-36-57 --ckpt_filename checkpoint0139.pth --output_dir /home/xxx/Results/AMPL/FS_classifier --evaluation_method classifier --iter_num 1000
# 10000 Test Acc at 100= 78.99% +- 0.18%
# AMPL 10000 Test Acc at 100= 90.61% +- 0.13%
python eval_ampl.py --server fs --num_shots 1 --ckp_path /home/xxx/Results/AMPL/CIFARFS84_phase2_n1_2023-11-13_00-04-47 --ckpt_filename checkpoint0005.pth --output_dir /home/xxx/Results/AMPL/FS_classifier05 --evaluation_method classifier --iter_num 1000
python eval_ampl.py --server fs --num_shots 5 --ckp_path /home/xxx/Results/AMPL/CIFARFS84_phase2_n1_2023-11-13_00-04-47 --ckpt_filename checkpoint0005.pth --output_dir /home/xxx/Results/AMPL/FS_classifier05 --evaluation_method classifier --iter_num 1000







