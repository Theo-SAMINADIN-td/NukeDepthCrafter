import gc
import os
import nuke
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
from diffusers.training_utils import set_seed
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.utils import save_video, read_video_frames
 

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        process_length: int = 195,
        overlap: int = 25,
        height : int = 1080,
        width : int = 1920,
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
        video_export: bool = False,
        dataset: str = "open"
    ):
        set_seed(seed)

        frames, target_fps = read_video_frames(
            video, process_length, target_fps, dataset=dataset,
        )
        
        print(f"==> video name: {video}, frames shape: {frames.shape}")
        
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
                
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        
        # save the depth map and visualization with the target FPS
        save_path = save_folder
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)
        
        
        if  video_export :
            save_video(res, save_path, fps=target_fps, video_export= video_export, output_height=height, output_width=width)
            nuke.createNode('Read')
            
            nuke.selectedNode().knob('file').setValue(r"%s" % (save_path))
            return [
                
                
                save_path + "_depth.mp4",
            ]
        else :
            save_video(res, save_path, fps=target_fps, video_export= video_export, output_height=height, output_width=width)
            try :
                nuke.createNode('Read')
                if "%04d" in save_folder :
                    nuke.selectedNode().knob('file').setValue(r"%s" % str(os.path.dirname(save_path)) +"/"+
                                                                            str(os.path.splitext(os.path.basename(save_path.replace('%04d', '####')))[0]) +".exr")
                elif "%03d" in save_folder :
                    nuke.selectedNode().knob('file').setValue(r"%s" % str(os.path.dirname(save_path))+"/"+
                                                                            str(os.path.splitext(os.path.basename(save_path.replace('%03d', '###')))[0]) +".exr")
                nuke.selectedNode().knob('first').setValue(1)
                nuke.selectedNode().knob('last').setValue(int(nuke.root().knob('last_frame').getValue())) 
            except :
                TypeError('Error creating read node')
                

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        height=1080,
        width=1920,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            height=height,
            width=width,
            process_length=process_length,
            
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        return res_path[:2]



