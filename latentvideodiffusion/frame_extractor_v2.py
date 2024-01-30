import cv2
import jax
import numpy as np
import os
import csv
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

class FrameExtractor:
    def __init__(self, directory_path, batch_size, key, target_size=(512,300)):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.video_gbl_idxs = np.zeros(len(self.video_files)) #holds global idx value for every video 
        self.total_frames = 0
        i = 0

        with open(os.path.join(directory_path, "data.csv"), newline='') as f:
            reader = csv.reader(f)
            self.data = list(reader)

        for row in self.data:    # Skip the header row and convert first values to integers
            row[1] = int(row[1])

        print(self.data)
        print(len(self.data))
        print(min(self.data, key=lambda x:x[1]))
        

        # video_data = []
        for f in self.video_files:
            frame_count = int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames += frame_count
            self.video_gbl_idxs[i] = self.total_frames
            i += 1
            # data = [f, frame_count]
            # video_data.append(data)
        self.cap = None
        self.target_size = target_size

        # with open('videodata.csv', 'w') as f:
        #     # Create a CSV writer object that will write to the file 'f'
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerows(video_data)



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self
    
    # # Original
    # def __next__(self):
    #     self.key, idx_key = jax.random.split(self.key)
    #     idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames)
    #     local_idx = 0
    #     video_idx = 0
    #     frames = []
        
    #     for global_idx in idx_array:
    #         if(global_idx < self.video_gbl_idxs[0]):
    #             local_idx = int(global_idx)
    #             #frame from video 0
    #         else:
    #             video_idx = np.searchsorted(self.video_gbl_idxs, int(global_idx))
    #             local_idx = int(global_idx) - int(self.video_gbl_idxs[video_idx-1])
    #         # print("frame", local_idx)
    #         # print("video", video_idx) 
    #         self.cap = cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx]))
    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
    #         ret, frame = self.cap.read()
    #         self.cap.release()

    #         if ret:
    #             frames.append(frame)

    #     array = jax.numpy.array(frames)
    #     return array.transpose(0,3,2,1)

    # def __next__(self):
    #     self.key, idx_key, *keys = jax.random.split(self.key, self.batch_size + 2)
    #     idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, len(self.data))

    #     video_paths = []
    #     frame_idxs = []
    #     for key, idx in zip(keys, idx_array):
    #         frame_idx = int(jax.random.randint(key, shape=(), minval=0, maxval=self.data[idx][1]))

    #         frame_idxs.append(frame_idx)
    #         video_paths.append(self.data[idx][0])

    #     print(len(video_paths))
    #     print(len(frame_idxs))

    #     print(video_paths)
    #     print(frame_idxs)
        
    #     def process_frame(video_path, frame_idx):
    #         self.cap = cv2.VideoCapture(os.path.join(self.directory_path, video_path))
    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #         ret, frame = self.cap.read()
    #         self.cap.release()

    #         if ret:
    #             return frame
    #         else:
    #             return None


    #     # video_paths = [self.data[idx] for idx in id]
    #     # frame_counts = [row[1] for row in self.data]
    #     # frame_pos_array = [jax.random.randint(key, shape=(), minval=0, maxval=frame_count) for key, frame_count in zip(keys, frame_counts)]
            
    #     l  = list(zip(video_paths, frame_idxs))
    #     print(len(l))
    #     print(l)

    #     with ProcessPoolExecutor() as executor:
    #         # results = [executor.submit(lambda p: process_frame(*p), [video_path, idx]) for video_path, idx in zip(video_paths, frame_idxs)]
    #         results = list(executor.map(lambda p: process_frame(*p), list(zip(video_paths, frame_idxs))))

    #     # for f in as_completed(results):
    #     #     if f.result() is not None: frames.append(f.result())

    #     frames = [result for result in results if result is not None]
    #     array = jax.numpy.array(frames)
    #     return array.transpose(0,3,2,1)
        

    #     # Create a pool of processes
    #     # with Pool() as pool:
    #     #     results = pool.starmap(self.process_frame, zip(video_paths, frame_idxs))

    #     # frames = [result for result in results if result is not None]
    #     # array = jax.numpy.array(frames)
    #     # return array.transpose(0,3,2,1)
    
    # Original
    def __next__(self):
        self.key, idx_key, *keys = jax.random.split(self.key, self.batch_size + 2)
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, len(self.data))
        frames = []

        for i, idx in enumerate(idx_array):
            video_path, frame_count = self.data[idx]
            frame_idx = jax.random.randint(keys[i], shape=(), minval=0, maxval=frame_count)
            frame_idx = int(frame_idx)

            self.cap = cv2.VideoCapture(os.path.join(self.directory_path, video_path))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            self.cap.release()

            if ret:
                frames.append(frame)
        
        array = jax.numpy.array(frames)
        return array.transpose(0,3,2,1)
    
    # # With fallback
    # def __next__(self):
    #     self.key, idx_key = jax.random.split(self.key)
    #     idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames)
    #     local_idx = 0
    #     video_idx = 0
    #     frames = []
        
    #     for global_idx in idx_array:
    #         print("GLOBAL INDEX: ", self.video_gbl_idxs[0])
    #         if(global_idx < self.video_gbl_idxs[0]):
    #             local_idx = int(global_idx)
    #             #frame from video 0
    #         else:
    #             video_idx = np.searchsorted(self.video_gbl_idxs, int(global_idx))
    #             local_idx = int(global_idx) - int(self.video_gbl_idxs[video_idx-1])
    #         # print("frame", local_idx)
    #         # print("video", video_idx) 
    #         self.cap = cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx]))
    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
    #         ret, frame = self.cap.read()
    #         while not ret:
    #             local_idx -= 1
    #             self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
    #             ret, frame = self.cap.read()
    #         self.cap.release()
    #         if ret:
    #             frames.append(frame)

    #     array = jax.numpy.array(frames)
    #     return array.transpose(0,3,2,1)
    

def extract_frames(video_path, num_frames, key, target_size=(512, 300)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(str(total_frames)+ " total frames")
    if num_frames > total_frames or num_frames <= 0:
        raise ValueError("Invalid number of frames specified.")

    random_indices = jax.random.randint(key, (num_frames,), 0, total_frames)

    frames = []
    for idx in random_indices:
        ret, frame = cap.read()
        if ret:
            # Resize video to specified target size
            # frame = cv2.resize(frame, target_size)
            frames.append(frame)

    cap.release()

    return jax.numpy.array(frames).transpose(0, 3, 2, 1)