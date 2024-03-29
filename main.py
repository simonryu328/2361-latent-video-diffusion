import argparse

import latentvideodiffusion as lvd
import latentvideodiffusion.utils
import latentvideodiffusion.vae
import latentvideodiffusion.diffusion
import latentvideodiffusion.plot

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Generate Visualizations using VAE and Diffusion Transformer.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the configuration file.')
    subparsers = parser.add_subparsers()

    # Training arguments for Diffusion Transformer
    train_diffusion_parser = subparsers.add_parser('train_diffusion')
    train_diffusion_parser.set_defaults(func=train_diffusion)
    train_diffusion_parser.add_argument('--checkpoint', type=str, default=None,
                                        help='Checkpoint iteration to load state from.')

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
    sample_diffusion_parser.add_argument('--name', type=str, required=False,
                                         default="diffusion_sample",
                                         help='[Optional] output file name to save as.')

    # Sampling arguments for VAE
    sample_vae_parser = subparsers.add_parser('sample_vae')
    sample_vae_parser.set_defaults(func=sample_vae)
    sample_vae_parser.add_argument('--checkpoint', type=str, required=True,
                                         help='Checkpoint iteration to load state from.')
    sample_vae_parser.add_argument('--name', type=str, required=False,
                                         default="vae_sample",
                                         help='[Optional] output file name to save as.')

    # Reconstructing arguments for VAE
    reconstruct_vae_parser = subparsers.add_parser('reconstruct_vae')
    reconstruct_vae_parser.set_defaults(func=reconstruct_vae)
    reconstruct_vae_parser.add_argument('--checkpoint', type=str, required=True,
                                         help='Checkpoint iteration to load state from.')
    reconstruct_vae_parser.add_argument('--name', type=str, required=False,
                                         default="vae_reconstruction",
                                         help='[Optional] output file name to save as.')

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
    
    # make video
    make_video_parser = subparsers.add_parser('make_video')
    make_video_parser.set_defaults(func=make_video)
    make_video_parser.add_argument('--data_dir', type=str, required=True,
                               help='The folder contaning image frames')
    make_video_parser.add_argument('--name', type=str, required=False,
                                         default="dt_video",
                                         help='[Optional] output file name to save as.')
       
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.func(args)

def train_vae(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.vae.train(args, cfg)

def train_diffusion(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.diffusion.train(args, cfg)

def sample_vae(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.vae.sample(args, cfg)

def reconstruct_vae(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.vae.reconstruct(args, cfg)

def sample_diffusion(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.diffusion.sample(args, cfg)

def encode_frames(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.utils.encode_frames(args, cfg)

def plot_loss(args):
    cfg = lvd.utils.load_config(args.config_file)
    lvd.plot.plot_loss(args, cfg)

def make_video(args):
    lvd.utils.make_video(args)

if __name__ == "__main__":
    main()


# def train_vae(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.vae.train(args, cfg)

# def train_diffusion(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.diffusion.train(args, cfg)

# def sample_vae(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.vae.sample(args, cfg)

# def sample_diffusion(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.diffusion.sample(args, cfg)

# def encode_frames(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.utils.encode_frames(args, cfg)

# def plot_loss(args):
#     cfg = lvd.utils.load_config(args.config_file)
#     lvd.plot.plot_loss(args, cfg)
