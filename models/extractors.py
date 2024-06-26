from .resnet import *
import timm
def build_extractor(c):
    if   c.extractor == 'resnet18':
        extractor = resnet18(pretrained=True, progress=True)
    elif c.extractor == 'resnet34':
        extractor = resnet34(pretrained=True, progress=True)
    elif c.extractor == 'resnet50':
        extractor = resnet50(pretrained=True, progress=True)
    elif c.extractor == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2':
        extractor = wide_resnet50_2(pretrained=True, progress=True)
    elif c.extractor == 'resnet101':
        extractor = timm.create_model(
                c.extractor,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
    elif c.extractor == 'convnextv2_base':
        print("================")
        extractor = timm.create_model(
            'convnext_base_384_in22ft1k',
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
    else:
        extractor = timm.create_model(
                c.extractor,
                pretrained=True,
                num_classes=0
            )
    output_channels = []
    # if 'wide' in c.extractor:
    #     for i in range(3):
    #         output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
    # else:
    #     for i in range(3):
    #         output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)
            
    print("Channels of extracted features:", output_channels)
    return extractor, output_channels