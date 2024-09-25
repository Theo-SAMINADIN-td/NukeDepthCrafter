from nukedc import UpdateBtn, CreateDCNode, GenerateDepthAction


m = nuke.menu('Nodes')
m = m.addCommand('DC', 'CreateDCNode()', tooltip="DepthCrafter")