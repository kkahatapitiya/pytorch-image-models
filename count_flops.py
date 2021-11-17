import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
#import torchvision.models as models
import timm.models.mlp_mixer as mm
import timm.models.vision_transformer as vit
import timm.models.swin_transformer as sw

inputs = (torch.randn((1,3,224,224)),)

#model = models.resnet18()
#model = mm.mixer_ti16_224()
#model = vit.vit_tiny_patch16_224()
#model = vit.cross_vit_tiny_patch16x32_224()
#model = pit.pit_ti_224(False)

#model = vit.vit_base_patch16_224()
#model = vit.vit_base_patch32_224()
#model = vit.vit_small_patch16_224()

#model = mm.mixer_s16_224()
model = mm.mixer_b32_224()

#model = sw.swin_tiny_patch4_window7_224()

#print(model)

flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops))
