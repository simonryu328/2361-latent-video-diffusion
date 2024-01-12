import argparse

from latentvideodiffusion  import utils,vae,diffusion,plot


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Generate Visualizations using VAE and Diffusion Transformer.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the configuration file.')
    subparsers = parser.add_subparsers()

    # Training arguments for Diffusion Transformer
    train_diffusion_parser = subparsers.add_parser('train_diffusion')
    train_diffusion_parser.set_defaults(func=train_diffusion)
    train_diffusion_parser.add_argument('--checkpoint', type=int, default=None,
                                        help='Checkpoint iteration to load state from.')
    train_diffusion_parser.add_argument('--data_dir', type=str, required=True,
                                        help='Directory path for Diffusion Transformer training data.')

    # Training arguments for VAE
    train_vae_parser = subparsers.add_parser('train_vae')
    train_vae_parser.set_defaults(func=train_vae)
    train_vae_parser.add_argument('--checkpoint', type=str, default=None,
                                  help='Checkpoint iteration to load state from.')

    # Sampling arguments for Diffusion Transformer
    sample_diffusion_parser = subparsers.add_parser('sample_diffusion')
    sample_diffusion_parser.set_defaults(func=sample_diffusion)
    sample_diffusion_parser.add_argument('--vae_checkpoint', type=str, required=True,
                                 help='VAE checkpoint iteration to load state from.')
    sample_diffusion_parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                                 help='Diffusion Transformer checkpoint iteration to load state from.')
    sample_diffusion_parser.add_argument('--data_dir', type=str, required=True,
                                 help='Directory with video latents')
    
    # Sampling arguments for VAE
    sample_vae_parser = subparsers.add_parser('sample_vae')
    sample_vae_parser.set_defaults(func=sample_vae)
    sample_vae_parser.add_argument('--checkpoint', type=str, required=True,
                                         help='Checkpoint iteration to load state from.')

    # Reconstructing arguments for VAE
    reconstruct_vae_parser = subparsers.add_parser('reconstruct_vae')
    reconstruct_vae_parser.set_defaults(func=reconstruct_vae)
    reconstruct_vae_parser.add_argument('--checkpoint', type=str, required=True,
                                         help='Checkpoint iteration to load state from.')
    reconstruct_vae_parser.add_argument('--data_dir', type=str, required=True,
                                         help='Directory with video')

    # Encoding arguments
    encode_parser = subparsers.add_parser('encode')
    encode_parser.set_defaults(func=encode_frames)
    encode_parser.add_argument('--vae_checkpoint', type=str, required=True,
                               help='VAE checkpoint iteration to load state from.')
    encode_parser.add_argument('--input_dir', type=str, required=True,
                               help='Directory path for input videos to be encoded.')
    encode_parser.add_argument('--output_dir', type=str, required=True,
                               help='Directory path to write encoded frames for Diffusion Transformer training.')

    # Loss Plotting arguments
    plot_loss_parser = subparsers.add_parser('plot_loss')
    plot_loss_parser.set_defaults(func=plot_loss)
    plot_loss_parser.add_argument('--type', type=str, required=True,
                               help='The loss to plot, can either be "vae" or "dt"')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.func(args)

def train_vae(args):
    cfg = utils.load_config(args.config_file)
    vae.train(args, cfg)

def train_diffusion(args):
    cfg = utils.load_config(args.config_file)
    diffusion.train(args, cfg)

def sample_vae(args):
    cfg = utils.load_config(args.config_file)
    vae.sample(args, cfg)

def reconstruct_vae(args):
    cfg = utils.load_config(args.config_file)
    vae.reconstruct(args, cfg)

def sample_diffusion(args):
    cfg = utils.load_config(args.config_file)
    diffusion.sample(args, cfg)

def encode_frames(args):
    cfg = utils.load_config(args.config_file)
    utils.encode_frames(args, cfg)

def plot_loss(args):
    cfg = utils.load_config(args.config_file)
    plot.plot_loss(args, cfg)

if __name__ == "__main__":
    main()
