from nukedc import UpdateBtn, CreateDCNode, GenerateDepthAction


m = nuke.menu('Nodes')
m = m.addCommand('DepthCrafter', 'CreateDCNode()', tooltip="DepthCrafter", icon="DEPTH_CRAFTER_ICON.png")