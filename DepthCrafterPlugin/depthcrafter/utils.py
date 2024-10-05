import numpy as np
import cv2
import matplotlib.cm as cm
import torch

dataset_res_dict = {
    "sintel":[448, 1024],
    "scannet":[640, 832],
    "kitti":[384, 1280],
    "bonn":[512, 640],
    "nyu":[448, 640],
}

def read_video_frames(video_path, process_length, target_fps, dataset):
    # a simple function to read video frames
    
    try :
        cap = cv2.VideoCapture(video_path)
    except :
        cap = cv2.VideoCapture(video_path, cv2.CAP_IMAGES)
    
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

    # # resize the video if the height or width is larger than max_res
    # if max(height, width) > max_res:
    #     scale = max_res / max(original_height, original_width)
    #     height = round(original_height * scale / 64) * 64
    #     width = round(original_width * scale / 64) * 64

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
    height, width = video_frames[0].shape[:2]
    is_color = video_frames[0].ndim == 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (width, height), isColor=is_color
    )
    frame_count = 1
    for frame in video_frames:
        frame = cv2.resize(frame, (output_width, output_height))
        if video_export :
            frame = (frame * 255).astype(np.uint8)
            if is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
            
            video_writer.release()
            return output_video_path
        
        else : 
            cv2.imwrite(r'%s' % output_video_path+"_depth"+"_"+str(f"{frame_count:04}")+".exr", frame.astype("float32"))
            frame_count += 1


class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
