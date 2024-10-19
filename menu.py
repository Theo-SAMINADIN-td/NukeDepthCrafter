from nukedc import CreateDCNode, UpdatePath, GenerateDepth


m = nuke.menu('Nodes')
m = m.addCommand('DepthCrafter', 'CreateDCNode()', tooltip="DepthCrafter", icon="DEPTH_CRAFTER_ICON.png")