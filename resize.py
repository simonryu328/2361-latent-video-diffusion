import os
import cv2

def resize_video(input_path, output_path, target_resolution=(512, 300)):
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, target_resolution)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, target_resolution)
        
        # Write the resized frame to the output video file
        out.write(resized_frame)
    
    # Release video capture and writer objects
    cap.release()
    out.release()
    
    cv2.destroyAllWindows()

def process_videos(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate through video files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith((".mp4", ".avi", ".mkv")):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            # Resize the video
            resize_video(input_path, output_path)

if __name__ == "__main__":
    input_directory = "../../data/"
    output_directory = "../../data/training_resize"
    
    process_videos(input_directory, output_directory)