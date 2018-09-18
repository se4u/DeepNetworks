#!/usr/bin/env python3
import os, sys, argparse, random
import numpy as np
sys.path.append('.')
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
sns.set(color_codes=True)
from argparse import Namespace
from tqdm import tnrange
import tensorflow as tf
from deep_networks import data_util
from deep_networks.models import blocks
from deep_networks.models.wgan import WGAN
PLOT_PFX = None
def plot_gaussian(
  ax,
  data,
  codes=None,
  color='w',
  size=5,
  color_palette=sns.color_palette('Set1', n_colors=8,  desat=.5)):
  ax.set_aspect('equal')
  ax.set_ylim((-size, size))
  ax.set_xlim((-size, size))
  ax.tick_params(labelsize=10)
  if codes is not None:
    c = [color_palette[i] for i in codes]
    color = None
  else:
    c = None
  sns.kdeplot(data[:, 0], data[:, 1],
              cmap='Blues', shade=True, shade_lowest=False, ax=ax)
  ax.scatter(data[:, 0], data[:, 1], linewidth=1, marker='+', c=c, color=color)

def get_gm_data(args):
  return data_util.gaussian_mixture(
    batch_size=args.gm_batch_size,
    scale=3.0,
    num_clusters=args.gm_num_classes)

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)

def sample_and_load(model, sample_step, checkpoint_dir, sample_fn):
  resume_step = None
  for step in sample_step:
    success, _ = model.load(checkpoint_dir, step)
    if success:
      sample_fn(model, step)
      resume_step = step
    else:
      break
  return resume_step

def sample_GAN(samples, num_batches, num_samples):
  def sample(gan, step):
    epoch = step // num_batches
    data = gan.sample(num_samples=num_samples)
    samples.append((epoch, data))
    # clear_output()
    f, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Epoch #{}'.format(epoch))
    plot_gaussian(ax, data)
    fn = PLOT_PFX.format(epoch=epoch)
    print('Writing', fn)
    plt.savefig(fn)
  return sample

def save_and_sample(checkpoint_dir, sample_fn):
  def sample(gan, step):
    # gan.save(checkpoint_dir, step)
    sample_fn(gan, step)
  return sample


def wgan_mog_conv():
  arg_parser = argparse.ArgumentParser(description='')
  arg_parser.add_argument('--gen_dim', default=300, type=int)
  arg_parser.add_argument('--dis_dim', default=300, type=int)
  arg_parser.add_argument('--num_epochs', default=3000, type=int)
  arg_parser.add_argument('--sample_interval', default=250, type=int)
  arg_parser.add_argument('--plot_pfx', default=None,
                          help='./wganexpt/fig/wgan.dim{dim}.epoch{epoch}.png')
  args=arg_parser.parse_args()
  plot_dirname = os.path.dirname(args.plot_pfx)
  print('Creating', plot_dirname)
  os.makedirs(plot_dirname, exist_ok=True)
  args.seed = 1234
  args.gm_num_examples = 5000
  args.gm_num_classes = 7
  args.gm_batch_size = 64
  args.gm_num_batches = args.gm_num_examples // args.gm_batch_size
  args.gm_output_shape = (2, )
  args.gm_log_dir = './wganexpt/logs/gm'
  args.gm_checkpoint_dir = './wganexpt/checkpoints/GM'
  args.sample_epochs = tuple(range(0, args.num_epochs, args.sample_interval))
  args.num_samples = 700
  args.gen_layers = 3
  args.dis_layers = 3
  set_seed(args)
  
  global PLOT_PFX
  PLOT_PFX = args.plot_pfx
  def myGeneratorFactory(inputs, output_shape, initializer, name,
                         reuse=False, activation_fn=None):
    return blocks.BasicGenerator(
      inputs=inputs,
      output_shape=output_shape,
      initializer=initializer,
      dim=args.gen_dim,
      num_layers=args.gen_layers,
      name=name)
  
  def myDiscriminatorFactory(inputs, input_shape, regularizer, initializer,
                             disc_activation_fn, reuse=False, name='discriminator'):
    return blocks.BasicDiscriminator(
      inputs=inputs,
      dim=args.dis_dim,
      num_layers=args.dis_layers,
      input_shape=input_shape,
      regularizer=regularizer,
      initializer=initializer,
      disc_activation_fn=disc_activation_fn,
      reuse=reuse,
      name=name)
  
  with tf.Graph().as_default():
    with tf.Session() as sess:
      samples = []
      sample_step = [i * args.gm_num_batches for i in args.sample_epochs]
      gm_data, gm_labels = get_gm_data(args)
      gan = WGAN(sess,
                 gm_data,
                 num_examples=args.gm_num_examples,
                 output_shape=args.gm_output_shape,
                 batch_size=args.gm_batch_size,
                 generator_cls=myGeneratorFactory,
                 discriminator_cls=myDiscriminatorFactory)
      gan._trange = tnrange
      gan.init_saver(tf.train.Saver(max_to_keep=None))
      sample_fn = sample_GAN(samples, args.gm_num_batches, args.num_samples)
      resume_step = sample_and_load(gan, sample_step, args.gm_checkpoint_dir, sample_fn)
      gan.train(num_epochs=args.num_epochs,
                log_dir=args.gm_log_dir,
                checkpoint_dir=args.gm_checkpoint_dir,
                resume_step=resume_step,
                sample_step=sample_step,
                save_step=None,
                sample_fn=save_and_sample(args.gm_checkpoint_dir, sample_fn))
      gm_samples['WGAN'] = samples
  return

if __name__ == '__main__':
  _fnc_=sys.argv[1]
  del sys.argv[1]
  globals()[_fnc_]()
