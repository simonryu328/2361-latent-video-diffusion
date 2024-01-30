import os
import cv2
import argparse

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
import optax

# import latentvideodiffusion as lvd
# import latentvideodiffusion.models.frame_vae as frame_vae
# import latentvideodiffusion.frame_extractor as fe
# import latentvideodiffusion.frame_transcode as ft

from . import utils, frame_extractor_v2 as fe, frame_transcode as ft
from .models import frame_vae 

import time

kl_a = 0.8 #TODO fix loading of this value

#Gaussian VAE primitives
def gaussian_kl_divergence(p, q):
    p_mean, p_log_var = p
    q_mean, q_log_var = q

    kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2
    return kl_div

def gaussian_log_probabilty(p, x):
    p_mean, p_log_var = p
    log_p = (-1/2)*((x-p_mean)**2/jnp.exp(p_log_var))-p_log_var/2-jnp.log(jnp.sqrt(2*jnp.pi))
    return log_p

def sample_gaussian(p, key):
    p_mean, p_log_var = p
    samples = jax.random.normal(key,shape=p_mean.shape)*jnp.exp(p_log_var/2)+p_mean
    return samples

def concat_probabilties(p_a, p_b):
    mean = jnp.concatenate([p_a[0],p_b[0]], axis=1)
    log_var = jnp.concatenate([p_a[1],p_b[1]], axis=1)
    return (mean, log_var)

@jax.jit
def vae_loss(vae, data, key):

    encoder, decoder = vae
    #Generate latent q distributions in z space
    q = jax.vmap(encoder)(data)

    #Sample Z values
    z = sample_gaussian(q, key)

    #Compute kl_loss terms
    z_prior = (0,0)
    kl = kl_a*gaussian_kl_divergence(q, z_prior)

    #Ground truth predictions
    p = jax.vmap(decoder)(z)

    #Compute the probablity of the data given the latent sample
    log_p = gaussian_log_probabilty(p, data)

    #Maximise p assigned to data, minimize KL div
    loss = sum(map(jnp.sum,[-log_p, kl]))/(data.size)

    return loss

def make_vae(n_latent, input_size, size_multipier, key):

    enc_key, dec_key = jax.random.split(key)

    e = frame_vae.VAEEncoder(n_latent, input_size, size_multipier, enc_key)
    d = frame_vae.VAEDecoder(n_latent, input_size, size_multipier, dec_key)
    
    vae = e,d
    return vae

def sample_vae(n_latent, n_samples, vae, key):
    z_key, x_key = jax.random.split(key)
    decoder = vae[1]
    p_z = (jnp.zeros((n_samples,n_latent)),)*2
    z = sample_gaussian(p_z, z_key)
    p_x = jax.vmap(decoder)(z)
    x = sample_gaussian(p_x, x_key)
    return x

def reconstruct_vae(n_samples, data_dir, vae, key):
    z_key, x_key = jax.random.split(key)
    encoder = vae[0]
    decoder = vae[1]
    print("Created Encoder and Decoder")
    extracted_frames = fe.extract_frames(data_dir, n_samples, x_key)
    print("Extracted Video Frames")

    encoded_frames = []
    data = jnp.array(extracted_frames, dtype=jnp.float32)
    mean, _ = jax.vmap(encoder)(data)
    encoded_frames.extend(mean)
    print("Encoded Frames")

    # z = sample_gaussian(encoded_frames, z_key)
    # p_x = jax.vmap(decoder)(z)
    # decoded_frames = sample_gaussian(p_x, x_key)
    decoded_frames = []
    for encoded_frame in encoded_frames:
        mean, _ = decoder(encoded_frame)
        frame = jax.lax.clamp(0., mean, 255.)
        decoded_frames.append(frame)
    print("Decoded Frames")
    return decoded_frames

def show_samples(samples):
    y = jax.lax.clamp(0., samples ,255.)
    frame = jnp.array(y.transpose(2,1,0),dtype=jnp.uint8)
    cv2.imshow('Random Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE model.')
    subparsers = parser.add_subparsers()
    
    #Training arguments
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint iteration to load state from.')
    
    #Sampling arguments
    sample_parser = subparsers.add_parser('sample')
    sample_parser.set_defaults(func=sample)
    sample_parser.add_argument('--checkpoint', type=int,
                        help='Checkpoint iteration to load state from.')
    
    sample_parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    return args

def sample(args, cfg):
    n_samples = cfg["vae"]["sample"]["n_sample"]
    n_latent = cfg["lvm"]["n_latent"]

    state = utils.load_checkpoint(args.checkpoint)
    trained_vae = state[0]

    key = jax.random.PRNGKey(cfg["seed"])
    samples = sample_vae(n_latent, n_samples, trained_vae, key)
    utils.show_samples(samples, args.name)

def reconstruct(args, cfg):
    n_samples = cfg["vae"]["reconstruct"]["n_sample"]
    video_path = cfg["vae"]["reconstruct"]["video_file"]
    generation_path = cfg["vae"]["reconstruct"]["generation_path"]
    state = utils.load_checkpoint(args.checkpoint)
    trained_vae = state[0]

    key = jax.random.PRNGKey(cfg["seed"])
    samples = reconstruct_vae(n_samples, video_path, trained_vae, key)
    utils.show_samples(samples, generation_path ,args.name)

def train(args, cfg):
    print("Entered VAE Training Function")
    ckpt_dir = cfg["vae"]["train"]["ckpt_dir"]
    lr = cfg["vae"]["train"]["lr"]
    ckpt_interval = cfg["vae"]["train"]["ckpt_interval"]
    video_paths_train = cfg["vae"]["train"]["data_dir_train"]
    video_paths_val = cfg["vae"]["train"]["data_dir_val"]
    batch_size = cfg["vae"]["train"]["bs"]
    clip_norm = cfg["vae"]["train"]["clip_norm"]
    metrics_path = cfg["vae"]["train"]["metrics_path"]
    kl_a = cfg["vae"]["train"]["kl_alpha"]
    adam_optimizer = optax.adam(lr)
    optimizer = optax.chain(adam_optimizer, optax.zero_nans(), optax.clip_by_global_norm(clip_norm))
    
    if args.checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        init_key, state_key = jax.random.split(key)
        vae = make_vae(cfg["lvm"]["n_latent"], cfg["transcode"]["target_size"],cfg["vae"]["size_multiplier"], init_key)
        opt_state = optimizer.init(vae)
        i = 0
        state = vae, opt_state, state_key, i
    else:
        checkpoint_path = args.checkpoint
        state = utils.load_checkpoint(checkpoint_path)
    
    dir_name = os.path.dirname(metrics_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # ################### ORIGINAL TRAINING LOOP #####################
    # with open(metrics_path,"a") as f:
    #     #TODO: Fix Frame extractor rng
    #     with fe.FrameExtractor(video_paths_train, batch_size, state[2]) as train_fe:
    #         with fe.FrameExtractor(video_paths_val, batch_size, state[2]) as val_fe:
    #             for _ in utils.tqdm_inf():
    #                 # Process input data
    #                 #loop_time = time.time()
    #                 train_data = jnp.array(next(train_fe), dtype=jnp.float32)
    #                 val_data = jnp.array(next(val_fe), dtype=jnp.float32)
    #                 #print("Processed input data in ", time.time()-loop_time, " seconds")
    #                 #loop_time = time.time()
    #                 # Update state
    #                 val_loss, _ = utils.update_state(state, val_data, optimizer, vae_loss)
    #                 train_loss, state = utils.update_state(state, train_data, optimizer, vae_loss)
    #                 #print("Updated state in ", time.time()-loop_time, " seconds")
    #                 # Print or log training and validation losses
    #                 #print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")

    #                 # Save metrics to file
    #                 f.write(f"{train_loss}\t{val_loss}\n")
    #                 f.flush()
    #                 iteration = state[3]
    #                 if (iteration % ckpt_interval) == (ckpt_interval - 1):
    #                     ckpt_path = utils.ckpt_path(ckpt_dir, iteration+1, "vae")
    #                     utils.save_checkpoint(state, ckpt_path)
    #                     print("---------CHECKPOINT SAVED----------")

    ################ REVISED TRAINING LOOP WITH DATA PARALLELISM ###############

    # Checking the number of devices we have here
    num_devices = len(jax.local_devices())
    print(f"Running on {num_devices} devices")

    # Get the device array
    devices = mesh_utils.create_device_mesh((num_devices,)).reshape(4,2)
    print(f"Device Array: \n\n{devices}\n")

    # Create a mesh from the device array
    mesh = Mesh(devices, axis_names=("ax1", "ax2"))

    # Define sharding with a partiton spec
    sharding = NamedSharding(mesh, P("ax1", None, "ax2", None))

    print(f"Number of logical devices: {len(devices)}")
    print(f"Shape of device array    : {devices.shape}")
    print(f"\nMesh     : {mesh}")
    print(f"Sharding : {sharding}\n\n")

    with open(metrics_path,"a") as f:
        #TODO: Fix Frame extractor rng
        with fe.FrameExtractor(video_paths_train, batch_size, state[2]) as train_fe:
            # with fe.FrameExtractor(video_paths_val, batch_size, state[2]) as val_fe:
                for _ in utils.tqdm_inf():
                    # Training iteration
                    train_data = jnp.array(next(train_fe), dtype=jnp.float32)
                    print(train_data.shape)

                    # train_data = jax.device_put(train_data, sharding)
                    # # print(f"Shard shape: {sharding.shard_shape(train_data.shape)}")  
                    # train_loss, state = utils.update_state(state, train_data, optimizer, vae_loss)

                    # # Validation iteration
                    # val_data = jnp.array(next(val_fe), dtype=jnp.float32)
                    # val_data = jax.device_put(val_data, sharding)
                    # val_loss, _ = utils.update_state(state, val_data, optimizer, vae_loss)

                    # # Print or log training and validation losses
                    # #print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")

                    # # Save metrics to file
                    # f.write(f"{train_loss}\t{val_loss}\n")
                    # f.flush()
                    # iteration = state[3]
                    # if (iteration % ckpt_interval) == (ckpt_interval - 1):
                    #     ckpt_path = utils.ckpt_path(ckpt_dir, iteration+1, "vae")
                    #     utils.save_checkpoint(state, ckpt_path)
                    #     print("---------CHECKPOINT SAVED----------")