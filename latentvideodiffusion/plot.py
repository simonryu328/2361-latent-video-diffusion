import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_loss(args, cfg):
    if args.type == "vae":
        metrics_path = cfg["vae"]["train"]["metrics_path"]
    elif args.type == "dt":
        metrics_path = cfg["dt"]["train"]["metrics_path"]
    else:
        print("Invalid type specified")
    plot_data_and_filtered(metrics_path, cfg)
#TODO add table settings for DT
def plot_data_and_filtered(file_path, cfg):
    # Read the data from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the data into two lists: train_loss and val_loss
    train_loss, val_loss = zip(*[map(float, line.strip().split('\t')) for line in lines])

    # Create a Butterworth low pass filter for train_loss
    b_train, a_train = butter(3, 0.02, btype='low', analog=False)
    y_train = filtfilt(b_train, a_train, train_loss)
    if val_loss:
        # Create a Butterworth low pass filter for val_loss
        b_val, a_val = butter(3, 0.02, btype='low', analog=False)
        y_val = filtfilt(b_val, a_val, val_loss)

    # Plot the original data and the filtered data on the same plot
    plt.figure(figsize=(16, 8))
    plt.loglog(train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.loglog(val_loss, 'r-', label='Validation Loss')
    plt.loglog(y_train, 'orange', linewidth=2, label='Filtered Training Loss')
    if val_loss:
        plt.loglog(y_val, 'green', linewidth=2, label='Filtered Validation Loss')
    plt.title('Loss vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (nats/dim)')

    cellText = [["Run", file_path[5:-4]],
                ["BatchSize", str(cfg["vae"]["train"]["bs"])],
                ["Learning Rate", str(cfg["vae"]["train"]["lr"])],
                ["KL Factor", str(cfg["vae"]["train"]["kl_alpha"])],
                ["Size Multiplier",str(cfg["vae"]["size_multiplier"])],
                ["Clip Norm", str(cfg["vae"]["train"]["clip_norm"])],
                ["N Latent",str(cfg["lvm"]["n_latent"])],
                ["Training Loss", str(round(y_train[-1], 5))],
                ["Valiation Loss", str(round(y_val[-1], 5))]]
    table = plt.table(cellText=cellText, cellLoc="left", bbox=[1,0, 0.2, 1])
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.subplots_adjust(right=0.8, left=0.1)
    file_name = file_path[:-4] + ".png"
    plt.savefig(file_name)
    print("Plot saved as ", file_name)