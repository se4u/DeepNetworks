FNC=${1-wgan}
DEBUG=${2-echo}

wgan () {
  for dim in 300 600 900 1200 1500 1800 2100; do
    $DEBUG rm -rf wganexpt/checkpoints wganexpt/logs
    CUDA_VISIBLE_DEVICES=$(free-gpu) $DEBUG ./expt.py wgan_mog_conv --num_epochs 10000 --sample_interval 250 --dis_dim $dim --plot_pfx "./wganexpt/fig/wgan.dim$dim.epoch{epoch}.png"
  done
}

wganqsub () {
  for dim in 300 600 900 1200 1500 1800 2100; do
	  $DEBUG qsub -V -j y -l gpu=1 -cwd ./qsub.sh $dim
  done
}

gif () {
    scp clsp:'/export/b20/prastog3/DeepNetworks/wganexpt/fig/wgan.dim300.epoch*.png' ~/Downloads/wganexpt/
    convert $(for e in {1..39}; do printf "~/Downloads/wganexpt/wgan.dim300.epoch$(( 250 * e )).png " ; done) ~/Downloads/wganexpt/tmp.gif
}

$FNC
