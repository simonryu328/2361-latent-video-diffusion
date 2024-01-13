import json
import tqdm
import os
import pickle
import dill
import functools
import numpy
import cv2
import jax
import equinox as eqx
import numpy as np
import jax.numpy as jnp


def ckpt_path(ckpt_dir,iteration, ckpt_type):
    filename = f'checkpoint_{ckpt_type}_{iteration}.pkl'
    ckpt_path = os.path.join(ckpt_dir, filename)
    return ckpt_path 


def save_checkpoint(state, filepath):
    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(state, f) #

def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    return state

def show_samples(samples, generation_path, name="r"):
    i = 0
    for x in samples:
        y = jax.lax.clamp(0., x ,255.)
        frame = np.array(y.transpose(2,1,0),dtype=np.uint8)
        file_path = os.path.join(generation_path, name + "_" + str(i) + ".jpg")
        print(file_path)
        if cv2.imwrite(file_path, frame):
            print("saved video")
        i = i + 1
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
def make_video(args):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Sort the image files to ensure correct order in the video
    image_files.sort()

    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(args.data_dir, image_files[0]))
    height, width, _ = first_image.shape
    print(height)
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    video_writer = cv2.VideoWriter(os.path.join(args.data_dir, args.name + ".mp4"), fourcc, 3, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(args.data_dir, image_file)
        print(image_path)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video saved at: {args.data_dir}")

def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        raise Exception("Config file not found.")
    except json.JSONDecodeError:
        raise Exception("Error decoding JSON in the config file.")

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key, i = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    i = i+1
    
    new_state = new_model, new_opt_state, new_key, i
    
    return loss,new_state

def tqdm_inf():
    def g():
      while True:
        yield
    return tqdm.tqdm(g())
        
def encode_frames(args, cfg):
    input_directory = args.input_dir
    output_directory = args.output_dir
    vae_checkpoint_path = args.vae_checkpoint

    def encode_frame(encoder, frame):
        frame = frame.transpose(2, 1, 0)
        encoded_frame = encoder(frame)
        return encoded_frame

    def encode_frames_batch(encoder, frames_batch):
        encoded_batch = jax.vmap(functools.partial(encode_frame, encoder))(frames_batch)
        return encoded_batch

    vae = load_checkpoint(vae_checkpoint_path)
    encoder = vae[0][0]

    video_files = [f for f in os.listdir(input_directory) if f.endswith(('.mp4', '.avi'))]

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for filename in video_files:
        file_base = os.path.splitext(filename)[0]
        vid_path = os.path.join(input_directory, filename)
        cap = cv2.VideoCapture(vid_path)

        # Initialize separate lists to hold original and encoded frames
        original_frames = []
        encoded_frames_1 = []
        encoded_frames_2 = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, cfg["transcode"]["target_size"])
            original_frames.append(frame)

            if len(original_frames) == cfg["transcode"]["bs"]:
                encoded_batch_1, encoded_batch_2 = encode_frames_batch(encoder, jnp.array(original_frames))
                
                encoded_frames_1.extend(encoded_batch_1.tolist())
                encoded_frames_2.extend(encoded_batch_2.tolist())

                original_frames.clear()

        cap.release()

        # Process any remaining frames
        if original_frames:
            encoded_batch_1, encoded_batch_2 = encode_frames_batch(encoder, jnp.array(original_frames))
            
            encoded_frames_1.extend(encoded_batch_1.tolist())
            encoded_frames_2.extend(encoded_batch_2.tolist())

        # Convert lists to NumPy arrays
        encoded_frames_array_1 = np.array(encoded_frames_1)
        encoded_frames_array_2 = np.array(encoded_frames_2)

        # Aggregate into a big tuple
        latents = (encoded_frames_array_1, encoded_frames_array_2)

        output_path = os.path.join(output_directory, f"{file_base}_encoded.pkl")

        # Save using pickle
        with open(output_path, 'wb') as f:
            pickle.dump(latents, f)

