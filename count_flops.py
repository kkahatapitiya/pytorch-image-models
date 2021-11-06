import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
#import torchvision.models as models
import timm.models.mlp_mixer as mm
import timm.models.vision_transformer as vit

inputs = (torch.randn((1,3,224,224)),)

#model = models.resnet18()
#model = mm.mixer_ti16_224()
model = vit.vit_tiny_patch16_224()
#model = vit.cross_vit_tiny_patch16x32_224()
#model = pit.pit_ti_224(False)

#print(model)

flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops))
