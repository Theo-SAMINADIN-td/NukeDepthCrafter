import nuke
from DepthCrafterPlugin.utils import *




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
    # Checking file name
    Output_path = nuke.thisNode().knob('OutputPath').getValue()
    if str(os.path.splitext(os.path.basename(Output_path)[0])) == '' :
        raise TypeError("Your must assign a file name")
    
    if ("%04d" not in Output_path and nuke.thisNode()['FileType'].value() == "exr")  :       
        raise TypeError("Your file must contains '####' or '###'")
    
    
    
    if os.path.exists(Output_path) :
       if not nuke.ask("Overwrite existing "+ Output_path +" ?") :
         nuke.thisNode()['OutputPath'].setValue("")
        
    if nuke.ask('<h3> Your generation settings : </h3>'+ "\n" 
                + "<hr class='solid'>"+ "\n" 
                "<b>Input Path : </b>" + str(nuke.thisNode().knob('FilePath').getValue()) + "\n"
                + "<hr class='solid'>"+ "\n" 
                "<b>Inference Steps: </b>" + str(int(nuke.thisNode().knob('InferSteps').value())) + "\n"
                "<b>Guidance Scale: </b>" + str(nuke.thisNode().knob('CFG').value()) + "\n"
                "<b>Number of frames: </b>" + str(nuke.thisNode().knob('FrameNumber').value()) + "\n"
                + "<hr class='solid'>"+ "\n" 
                "<b>Output Height: </b>" + str(int(nuke.thisNode().knob('Height').value())) + "\n"
                "<b>Output Width: </b>" + str(int(nuke.thisNode().knob('Width').value())) +  "\n"
                + "<hr class='solid'>"+ "\n" 
                "<b>Targeted FPS: </b>" + str(nuke.thisNode().knob('FPS').value()) + "\n"
                + "<hr class='solid'>"+ "\n" 
                "<b>Output file type: </b>" + str(nuke.thisNode().knob('FileType').value()) + "\n"
                "<b>Dataset: </b>" + str(nuke.thisNode().knob('Dataset_Select').value()) +"\n"
                + "<hr class='solid'>"+ "\n" 
                "<b>Output Path:  </b>" + str(nuke.thisNode().knob('OutputPath').getValue()) + "\n"
                + "<hr class='solid'>" + "\n"+ 
                "<h3 align='right'> <font size='3'>Launch generation ? </h3>"
    
    ):                      
            if (nuke.thisNode().knob('FileType').value() == "mp4") :
                VideoExportBool = 1
            else :
                VideoExportBool = 0
            depthcrafter_demo = DepthCrafterDemo(
                unet_path=os.path.expandvars(r"C:\Users\$USERNAME\.nuke\DepthCrafterPlugin"),
                pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                cpu_offload=nuke.thisNode().knob('CPUOFF_OPT').value(),
            )
            # process the video
            video_paths = [nuke.thisNode().knob('FilePath').getValue()]
            
            for video in video_paths:
                depthcrafter_demo.infer(
                    video,
                    int(nuke.thisNode().knob('InferSteps').value()),        # args.num_inference_steps,
                    nuke.thisNode().knob('CFG').value(),                    # args.guidance_scale,
                    nuke.thisNode().knob('OutputPath').getValue(),          #args.save_folder,
                    window_size= 110,                                       #args.window_size,
                    process_length=nuke.thisNode().knob('FrameNumber').value(),    #args.process_length,
                    overlap= 25,                                            #args.overlap,
                    height=int(nuke.thisNode().knob('Height').value()),
                    width= int(nuke.thisNode().knob('Width').value()),      #args.max_res,
                    target_fps=nuke.thisNode().knob('FPS').value(),         #args.target_fps,
                    seed= 42,                                               #args.seed,
                    track_time= False,                                      #args.track_time,
                    save_npz= False,                                        #args.save_npz,
                    video_export=VideoExportBool,
                    dataset=nuke.thisNode().knob('Dataset_Select').value(),
                )
                
                # clear the cache for the next video
                gc.collect()
                torch.cuda.empty_cache()
                nuke.message('Generating depth for ' + video + ' Done')




def CreateDCNode():
    getInputInfos()
    nuke.createNode('NoOp')
    s = nuke.selectedNode()
    s.knob('name').setValue('DepthCrafter')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdateBtn()' ))
   
    s.addKnob(nuke.Text_Knob(''))

    s.addKnob(nuke.Enumeration_Knob('CPUOFF_OPT', 'CPU Offload Options', ['model', 'sequential', 'none']))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Int_Knob("InferSteps", 'Inference Steps'))
    s.addKnob(nuke.Double_Knob("CFG", 'Guidance scale'))
    s.addKnob(nuke.Int_Knob("FrameNumber", 'Number of frame'))
    s.addKnob(nuke.Int_Knob("Height", 'Height'))
    s.addKnob(nuke.Int_Knob("Width", 'Width'))
    s.addKnob(nuke.Enumeration_Knob('Dataset_Select', 'Dataset', ["open","sintel","scannet","kitti","bonn","nyu"]))
    
    s.addKnob(nuke.Text_Knob(' ', ''))
   
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateDepth', 'Generate Depth', 'GenerateDepthAction()'))
    
    
    
    
### SETTING RANGES, DEFAULT VALUES, TOOLTIP & FORMATING ###
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue())) #ADD ROOT FPS BY DEFAULT
    s['InferSteps'].setValue(25)
    s['CFG'].setValue(1.2)
    s['FrameNumber'].setValue(int(getInputInfos.FrameNumber))
    s['Height'].setValue(1080)
    s['Width'].setValue(1920)

    s['InferSteps'].setRange(1, 40)
    s['CFG'].setRange(1, 20)
    

    s['CPUOFF_OPT'].setFlag(nuke.STARTLINE)
    s['FPS'].setFlag(nuke.STARTLINE)
    s['InferSteps'].setFlag(nuke.STARTLINE)
    s['CFG'].setFlag(nuke.STARTLINE)
    s['FrameNumber'].setFlag(nuke.STARTLINE)
    s['Height'].setFlag(nuke.STARTLINE)
    s['Width'].setFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateDepth'].setFlag(nuke.STARTLINE)
    s['Dataset_Select'].setFlag(nuke.STARTLINE)
    
    s['CPUOFF_OPT'].setTooltip("To save memory, we can offload the model to CPU. Model is the default one, Sequential will be slower but save more memory")
    s['FPS'].setTooltip("Target FPS for the output video")
    s['InferSteps'].setTooltip("Number of inference steps")
    s['CFG'].setTooltip("Guidance scale/CFG")
    s['FrameNumber'].setTooltip("Number of frame to generate")
    s['Height'].setTooltip("Video output height")
    s['Width'].setTooltip("Video output width")
    s['OutputPath'].setTooltip("path/to/your/file.ext to create a image sequence add #### or ### ")
    s['GenerateDepth'].setTooltip("Generate Depth")
    s['Dataset_Select'].setTooltip("Select the Dataset Resolution which your generation will be generate from")


