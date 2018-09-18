FNC=${1-wgan}
DEBUG=${2-echo}

wgan () {
  for dim in 300 600 900 1200 1500 1800 2100; do
    CUDA_VISIBLE_DEVICES=$(free-gpu) ./expt.py wgan_mog_conv --num_epochs 10000 --sample_interval 250 --dis_dim $dim
  done
}

wganqsub () {
    for dim in 300 600 900 1200 1500 1800 2100; do
	$DEBUG qsub -V -j y -l gpu=1 -cwd ./qsub.sh $dim
    done
}

$FNC
