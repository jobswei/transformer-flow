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
    elif "convnext" in c.extractor:
        print("================")
        extractor = timm.create_model(
            c.extractor,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
    elif c.extractor=="inceptionnext":
        pretrained_cfg_overlay = {'file' : r"/home/xiaomi/.cache/huggingface/hub/models--timm--inception_next_base.sail_in1k_384/snapshots/fca9182b12aeeb83f0bec857d7e7db24ecf414c0/model.safetensors"}
        extractor=timm.create_model(
            "inception_next_base.sail_in1k_384",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
            pretrained_cfg_overlay=pretrained_cfg_overlay
        )
    elif c.extractor=="rdnet":
        pretrained_cfg_overlay = {'file' : r"/home/xiaomi/.cache/huggingface/hub/models--naver-ai--rdnet_base.nv_in1k/snapshots/88e964238b391484ccffc309216151c3f22de527/pytorch_model.bin"}
        extractor=timm.create_model(
            "rdnet_base.nv_in1k",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
            pretrained_cfg_overlay=pretrained_cfg_overlay
        )
    elif c.extractor=="ghostnet":
        pretrained_cfg_overlay = {'file' : r"/home/xiaomi/.cache/huggingface/hub/models--timm--ghostnetv2_160.in1k/snapshots/f29f8843944b3f614fbea96f3847e88080f9ed27/model.safetensors"}
        extractor=timm.create_model(
            "ghostnetv2_160.in1k",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
            pretrained_cfg_overlay=pretrained_cfg_overlay
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
            
    # print("Channels of extracted features:", output_channels)
    return extractor, output_channels
