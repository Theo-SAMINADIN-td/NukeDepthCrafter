import nuke
from DepthCrafterPlugin.utils import *
from diffusers.training_utils import set_seed

def getInputInfos():
    f = nuke.thisNode().dependencies()
    
    try :

        for i in f:
            getInputInfos.read = i


        try :
            getInputInfos.FrameNumber = getInputInfos.read.knob('last').getValue()
        except : 
            getInputInfos.FrameNumber = nuke.root().knob('last_frame').getValue()


        try :
            getInputInfos.path = getInputInfos.read.knob('file').getValue()
        except : 
            getInputInfos.path = ''
    except : 
        getInputInfos.FrameNumber = nuke.root().knob('last_frame').getValue()
        getInputInfos.path = ''
    
    

        





def UpdateBtn():
    getInputInfos()

    FilePath = getInputInfos.path
    
    nuke.thisNode().knob('FilePath').setValue(FilePath)


def GenerateDepthAction():
    
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=r"C:\Users\Theo\.nuke\DepthCrafterPlugin",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload=nuke.thisNode().knob('CPUOFF_OPT').value(),
    )
    # process the videos, the video paths are separated by comma
    video_paths = [nuke.thisNode().knob('FilePath').getValue()]
    #args.video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
            video,
            nuke.thisNode().knob('InferSteps').value(),                    # args.num_inference_steps,
            nuke.thisNode().knob('CFG').value(),                    # args.guidance_scale,
            nuke.thisNode().knob('OutputPath').getValue(),       #args.save_folder,
            window_size= 110,       #args.window_size,
            process_length=nuke.thisNode().knob('FrameNumber').value(),    #args.process_length,
            overlap= 25,              #args.overlap,
            max_res=nuke.thisNode().knob('MaxRes').value(),              #args.max_res,
            target_fps=nuke.thisNode().knob('FPS').value(),           #args.target_fps,
            seed= 42  ,                 #args.seed,
            track_time= False   ,    #args.track_time,
            save_npz= True,          #args.save_npz,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        nuke.message('Render'+ video +' Done')



def CreateDCNode():
    getInputInfos()
    nuke.createNode('NoOp')
    s = nuke.selectedNode()
    s.knob('name').setValue('DepthCrafter2')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdateBtn()' ))
   
    s.addKnob(nuke.Text_Knob(''))

    s.addKnob(nuke.Enumeration_Knob('CPUOFF_OPT', 'CPU Offload Options', ['model', 'sequential', 'none']))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Int_Knob("InferSteps", 'Inference Steps')) #NEED TO CREATE FUNCTION TO ROUND UP
    s.addKnob(nuke.Double_Knob("CFG", 'Guidance scale'))
    s.addKnob(nuke.Int_Knob("FrameNumber", 'Number of frame'))
    s.addKnob(nuke.Int_Knob("MaxRes", 'Maximum Resolution'))

    s.addKnob(nuke.Text_Knob(' ', ''))

    s.addKnob(nuke.PyScript_Knob('GenerateDepth', 'Generate Depth', 'GenerateDepthAction()'))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    
    # s.addKnob(nuke.Array_Knob("HeightInput", 'Height'))
    # s.addKnob(nuke.Array_Knob("WidthtInput", 'Width'))
    # s.addKnob(nuke.EvalString_Knob("OutputName", 'Output Name', ''))
    # S.addKnob(nuke.Format_Knob("OutputFormat", 'OUtput Format'))
    # s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['.exr', '.png', '.tiff']))
    
### SETTING RANGES, DEFAULT VALUES & FORMATING ###
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue())) #ADD ROOT FPS BY DEFAULT
    s['InferSteps'].setValue(25)
    s['CFG'].setValue(1.2)
    s['FrameNumber'].setValue(int(getInputInfos.FrameNumber))
    s['MaxRes'].setValue(1024)

    s['InferSteps'].setRange(1, 40)
    s['CFG'].setRange(1, 20)
    

    s['CPUOFF_OPT'].setFlag(nuke.STARTLINE)
    s['FPS'].setFlag(nuke.STARTLINE)
    s['InferSteps'].setFlag(nuke.STARTLINE)
    s['CFG'].setFlag(nuke.STARTLINE)
    s['FrameNumber'].setFlag(nuke.STARTLINE)
    s['MaxRes'].setFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateDepth'].setFlag(nuke.STARTLINE)

    print(nuke.thisNode().allKnobs())

