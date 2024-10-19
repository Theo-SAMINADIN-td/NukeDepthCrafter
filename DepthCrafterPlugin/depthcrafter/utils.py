import numpy as np
import cv2
import os
import nuke

dataset_res_dict = {
    "sintel":[448, 1024],
    "scannet":[640, 832],
    "kitti":[384, 1280],
    "bonn":[512, 640],
    "nyu":[448, 640],
}


video_extensions = {'.mp4', '.mov',}
img_extensions = { '.jpeg', '.jpg', '.png', '.tiff', '.tif'}


def read_video_frames(video_path, process_length, target_fps, dataset):
    # a simple function to read video frames
    nuke.tprint('Reading frames : ' +str(video_path) )
    
    
    if any(video_path.lower().endswith(ext) for ext in video_extensions) :
        cap = cv2.VideoCapture(video_path)
    elif any(video_path.lower().endswith(ext) for ext in img_extensions) :
        cap = cv2.VideoCapture(video_path, cv2.CAP_IMAGES)
    else :
        raise TypeError('Unsupported input format. Input must be: '+ str(video_extensions) + ' or ' + str(img_extensions))
    
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # round the height and width to the nearest multiple of 64
    if dataset=="open":        
        frame_height = round(original_height / 64) * 64
        frame_width = round(original_width / 64) * 64
    else:
        frame_height = dataset_res_dict[dataset][0]
        frame_width = dataset_res_dict[dataset][1]   

    if target_fps < 0:
        target_fps = original_fps

    stride = max(round(original_fps / target_fps), 1)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame.astype("float32") / 255.0)
            
       
            
       
        frame_count += 1
    cap.release()

    frames = np.array(frames)
    
    
    return frames, target_fps


def save_video(
    video_frames,
    output_video_path,
    fps: int = 15,
    video_export: bool = False,
    output_height: int = 1080,
    output_width: int = 1920
) -> str:
    # a simple function to save video frames
    nuke.tprint('Saving files...' )
    output_size = (output_width , output_height)
    
    output_video_path_folder = os.path.dirname(output_video_path) + "/"
    output_file_name = str(os.path.splitext(os.path.basename(output_video_path))[0])
    clean_path = os.path.join(output_video_path_folder, output_file_name)+".mp4"
    
    resized_vid = cv2.resize(video_frames[0], output_size, interpolation=cv2.INTER_AREA)
    height, width = resized_vid.shape[:2]
    is_color = video_frames[0].ndim == 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
       clean_path, fourcc, fps, (width, height), isColor=is_color
    )
    frame_count = 1
    
    
    nuke.tprint("Writing...")
    for frame in video_frames:
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA )
        if video_export :
            
            frame = (frame * 255).astype(np.uint8)
            if is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)           
        
        else : 
            
            try :     
                if "%04d" in output_video_path :
                    save_path = os.path.join(
                        
                        output_video_path_folder, output_file_name.replace('%04d', str(f"{frame_count:04}")))
                    
                if "%03d" in output_video_path :
                    save_path = os.path.join(
                    output_video_path_folder, output_file_name.replace('%03d', str(f"{frame_count:03}")))
            except :
                raise TypeError("Image sequence should be path/to/img_#### or path/to/img_###")
            cv2.imwrite(r'%s' % save_path +".exr", frame.astype("float32"))
        frame_count += 1
    
    nuke.tprint("Writing done")
    if video_export :
        video_writer.release() 
        return output_video_path
        
     