## ___***DepthCrafter for Nuke15***___
<div align="center">

[Depthcrafter by TENCENT](https://github.com/Tencent/DepthCrafter)
<br><br>
Adapted for Nuke by 
<br><br>
[Theo SAMINADIN](https://github.com/Theo-SAMINADIN-td)

</div>

## 🗒️ Requirements
According to [Nuke v15 Third-Party Libraries and Fonts](https://learn.foundry.com/nuke/content/misc/studio_third_party_libraries.html)

Python 3.10.10
<br><br>
## 🛠️ Installation
<br></br>
Clone this repo into your .nuke (by default C:\Users\\%UserProfile%\\.nuke) :
```bash
git clone https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter.git
```
<br>
OR
<br><br>
Download this repo as ZIP file and extract it in your .nuke

  <br><br>
   Your tree should look like that <br><br>
![Project file tree](https://github.com/user-attachments/assets/97a69ab8-bc80-4b43-a20b-ffe47e30da14)


<br></br>
Download the [Model from Tencent HF page](https://huggingface.co/tencent/DepthCrafter/blob/main/diffusion_pytorch_model.safetensors) and put it in C:\Users\\%UserProfile%\\.nuke\DepthCrafterPlugin

<br><br>

Install Dependencies :

According to [Nuke v15 Third-Party Libraries and Fonts](https://learn.foundry.com/nuke/content/misc/studio_third_party_libraries.html)
<br><br><br>
Pytorch/Cuda and xformers version
<br>
```bash
pip install torch==2.1.1+cu118 xformers --index-url https://download.pytorch.org/whl/cu118
```
Then
```bash
pip install -r requirements.txt
```
Refering to [requirements.txt](https://github.com/Theo-SAMINADIN-td/NukeDepthCrafter/blob/main/DepthCrafterPlugin/requirements.txt)

In init.py add your PATH. Check it by typing System Environment Variables in your start menu then click on Environment Variables and look for PATH.
```bash
nuke.pluginAddPath("path/to/your/PATH")
```
