CUDA_VISIBLE_DEVICES=$(free-gpu) ./expt.py wgan_mog_conv --num_epochs 10000 --sample_interval 250 --dis_dim $1 --plot_pfx "./wganexpt/fig/wgan.dim$1.epoch{epoch}.png"
