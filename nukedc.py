import nuke
import os
from DepthCrafterPlugin.utils import *
import threading
from DepthCrafterPlugin.depthcrafter.utils import video_extensions, img_extensions

nuke.tprint('Current thread : ' +str(threading.current_thread().name) )



class InputInfos :
    read = None
    FrameNumber = None
    path = None

    @classmethod
    def getInputInfos(cls):
        f = nuke.thisNode().dependencies()
        
        try :

            for i in f:
                cls.read = i


            try :
                cls.FrameNumber = cls.read.knob('last').getValue()
            except : 
                cls.FrameNumber = nuke.root().knob('last_frame').getValue()


            try :
                cls.path = cls.read.knob('file').getValue()
            except : 
                cls.path = ''
        except : 
            cls.FrameNumber = nuke.root().knob('last_frame').getValue()
            cls.path = ''
    
    
    

def UpdatePath():
    InputInfos.getInputInfos()

    FilePath = InputInfos.path
    
    nuke.thisNode().knob('FilePath').setValue(FilePath)



class GenerateDepth :
    unet_path = None
    pre_train_path = None
    cpu_offload = None
    video_paths = None
    infer_steps = None
    guid_scale = None
    Output_path = None
    process_length = None
    height = None
    width = None
    target_fps = None
    video_export = None
    dataset = None                
    frame_range = None               
                   
    def GenerateDepthAction():
        
        Output_path = nuke.thisNode().knob('OutputPath').getValue()
        # Checking file name
                    
        if str(os.path.splitext(os.path.basename(Output_path))[0]) == '' :
            
            raise TypeError("You must assign a file name")
            
        if any(nuke.thisNode()['FilePath'].value().lower().endswith(ext) for ext in video_extensions) == False and any(nuke.thisNode()['FilePath'].value().lower().endswith(ext) for ext in img_extensions) == False :
            
            raise TypeError('Unsupported input format. Input must be: '+ str(video_extensions) + ' or ' + str(img_extensions))
            

        if ("%04d"not in Output_path and nuke.thisNode()['FileType'].value() == "exr")  :       
            
            if ("%03d"not in Output_path)  :       
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
                if nuke.thisNode().knob('CPUOFF_OPT').value() == 'None':
                    cpu_opt = None
                else : 
                    cpu_opt = nuke.thisNode().knob('CPUOFF_OPT').value()
                depthcrafter_demo = DepthCrafterDemo(
                    unet_path=os.path.join(os.path.dirname(__file__), "DepthCrafterPlugin"),
                    pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                    cpu_offload=cpu_opt,
                )
                # process the video
                video_paths = [nuke.thisNode().knob('FilePath').getValue()]
                
                for video in video_paths:
                    video = video
                    num_denoising_steps = int(nuke.thisNode().knob('InferSteps').value())          # args.num_inference_steps,
                    guidance_scale = nuke.thisNode().knob('CFG').value()                        # args.guidance_scale,
                    save_folder = nuke.thisNode().knob('OutputPath').getValue()                 #args.save_folder,
                    window_size= 110                                                           #args.window_size,
                    process_length=nuke.thisNode().knob('FrameNumber').value()                  #args.process_length,
                    overlap= 25                                                                 #args.overlap,
                    height=int(nuke.thisNode().knob('Height').value())
                    width= int(nuke.thisNode().knob('Width').value())                          #args.max_res,
                    target_fps=nuke.thisNode().knob('FPS').value()                              #args.target_fps,
                    seed= 42                                             #args.seed,
                    track_time= False                                      #args.track_time,
                    save_npz= False                                       #args.save_npz,
                    video_export=VideoExportBool
                    dataset=nuke.thisNode().knob('Dataset_Select').value()
                    frame_range = [int(nuke.thisNode().knob('FrameRangeMin').value()) , int((nuke.thisNode().knob('FrameRangeMax').value()+ 1))]
                    
                    # create child thread on Inference mode
                    childThread = threading.Thread(target=depthcrafter_demo.infer, args=(
                        
                        video,
                        num_denoising_steps,       
                        guidance_scale,                 
                        save_folder,        
                        window_size,                                
                        process_length,   
                        overlap,                                            
                        height,
                        width,     
                        target_fps,       
                        seed,                                              
                        track_time,                                   
                        save_npz,                                       
                        video_export,
                        dataset,
                        frame_range
                        
                    ))
                    
                    # starting child thread
                    childThread.start()
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    



def CreateDCNode():
    InputInfos.getInputInfos()
    nuke.createNode('NoOp')
    s = nuke.selectedNode()
    s.knob('name').setValue('DepthCrafter')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()' ))
   
    s.addKnob(nuke.Text_Knob(''))

    s.addKnob(nuke.Enumeration_Knob('CPUOFF_OPT', 'CPU Offload Options', ['model', 'sequential', 'None']))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Int_Knob("InferSteps", 'Inference Steps'))
    s.addKnob(nuke.Double_Knob("CFG", 'Guidance scale'))
    s.addKnob(nuke.Int_Knob("FrameNumber", 'Number of frame'))
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Int_Knob("Height", 'Height'))
    s.addKnob(nuke.Int_Knob("Width", 'Width'))
    s.addKnob(nuke.Enumeration_Knob('Dataset_Select', 'Dataset', ["open","sintel","scannet","kitti","bonn","nyu"]))
    
    s.addKnob(nuke.Text_Knob(' ', ''))
   
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateDepth', 'Generate Depth', 'GenerateDepth.GenerateDepthAction()'))
    s.addKnob(nuke.nuke.PythonCustomKnob('KnobChanged', 'Knob Change', '''nuke.thisNode().knob("knobChanged").setValue("""

if "exr" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 

elif "png" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 

elif "tiff" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 

elif "tif" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 

elif "jpeg" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 
    
elif "jpg" in nuke.thisNode().knob("FilePath").value(): 

    nuke.thisNode().knob("FrameRangeMin").setEnabled(True)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(True)	
    nuke.thisNode().knob("FrameNumber").setEnabled(False) 

else :
    nuke.thisNode().knob("FrameNumber").setEnabled(True) 
    nuke.thisNode().knob("FrameRangeMin").setEnabled(False)
    nuke.thisNode().knob("FrameRangeMax").setEnabled(False)	
                                    
    """)
'''
    ) )
    
    
    
### SETTING RANGES, DEFAULT VALUES, TOOLTIP & FORMATING ###
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue())) #ADD ROOT FPS BY DEFAULT
    s['InferSteps'].setValue(25)
    s['CFG'].setValue(1.2)
    s['FrameNumber'].setValue(int(InputInfos.FrameNumber))
    s['Height'].setValue(1080)
    s['Width'].setValue(1920)

    s['InferSteps'].setRange(1, 40)
    s['CFG'].setRange(1, 20)
    

    s['CPUOFF_OPT'].setFlag(nuke.STARTLINE)
    s['FPS'].setFlag(nuke.STARTLINE)
    s['InferSteps'].setFlag(nuke.STARTLINE)
    s['CFG'].setFlag(nuke.STARTLINE)
    s['FrameNumber'].setFlag(nuke.STARTLINE)
    s['FrameRangeMax'].clearFlag(nuke.STARTLINE)
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
    s['Dataset_Select'].setTooltip("""Select the Dataset Resolution which your generation will be generate from 
                                   
sintel: 448x1024
scannet: 640x832
kitti: 384x1280
bonn: 512x640
nyu: 448x640
                                   """)


