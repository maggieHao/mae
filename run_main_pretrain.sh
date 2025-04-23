python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py  --batch_size 64 \
    --world_size 2 \
    --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --data_path '/home/maggie/Documents/ImageNet1k'\
    --blr 1.5e-4 --weight_decay 0.05
