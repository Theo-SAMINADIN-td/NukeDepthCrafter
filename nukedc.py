import nuke
from DepthCrafterPlugin.utils import *
from diffusers.training_utils import set_seed


def UpdateBtn():

    f = nuke.thisNode().dependencies()
    
    for i in f:
        read = i
        
    
    FilePath = read.knob('file').getValue()
    
    nuke.thisNode().knob('FilePath').setValue(FilePath)


    

def GenerateDepthAction():
    
    depthcrafter_demo = DepthCrafterDemo(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )
    # process the videos, the video paths are separated by comma
    video_paths = [nuke.thisNode().knob('FilePath').getValue()]
    #args.video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
            video,
            4,                    # args.num_inference_steps,
            1,                    # args.guidance_scale,
            nuke.thisNode().knob('OutputPath').getValue(),       #args.save_folder,
            window_size= 110,       #args.window_size,
            process_length= 195 ,       #args.process_length,
            overlap= 25,              #args.overlap,
            max_res= 512 ,              #args.max_res,
            target_fps= 10 ,           #args.target_fps,
            seed= 42  ,                 #args.seed,
            track_time= False   ,    #args.track_time,
            save_npz= True,          #args.save_npz,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        nuke.message('Render'+ video +' Done')



def CreateDCNode():
   
    nuke.createNode('NoOp')
    s = nuke.selectedNode()
    s.knob('name').setValue('DepthCrafter2')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdateBtn()' ))
    s.addKnob(nuke.Text_Knob(''))
    s.addKnob(nuke.PyScript_Knob('GenerateDepth', 'Generate Depth', 'GenerateDepthAction()'))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateDepth'].setFlag(nuke.STARTLINE)
    print(nuke.thisNode().allKnobs())

    

    


#m = nuke.menu('Nodes')
#m = m.addCommand('DC', 'DepthCrafterAction()')

