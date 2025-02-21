import torch
import torchvision
from torchvision import transforms

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        for param in self.parameters():
            param.requires_grad = False
        base_model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        self.modified_features = torch.nn.Sequential()
        for i, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.modified_features.append(torch.nn.AvgPool2d(kernel_size=2))
            elif isinstance(layer, torch.nn.Conv2d):
                self.modified_features.append(torch.nn.ReplicationPad2d(1))
                unpadded_conv = torch.nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=0, dilation=layer.dilation, groups=layer.groups, bias=(layer.bias is not None))
                unpadded_conv.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    unpadded_conv.bias.data = layer.bias.data.clone()
                self.modified_features.append(unpadded_conv)
                
            else:
                self.modified_features.append(layer)


        self.blocks = torch.nn.ModuleList()
        self.blocks.extend([self.modified_features[0:3],
                             self.modified_features[3:10],
                             self.modified_features[10:17],
                             self.modified_features[17:30]])
        
    def forward(self, block, x):
        for layer in self.blocks[block]:
            if isinstance(layer, torch.nn.AvgPool2d):
                skiplayer = x
                x = layer(x)
                skiplayer = skiplayer - torch.nn.functional.interpolate(x, size=skiplayer.shape[2:], mode="bilinear", align_corners=False)
            else:
                x = layer(x)
        
        if block == 0:
            return x
        else:
            return x, skiplayer
        
class DecoderBlock(torch.nn.Module):
    def __init__(self, layers, input_channels=0, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.layers = torch.nn.Sequential(*layers)
        self.input_channels = input_channels
        self.skip_channels = skip_channels
    
    def forward(self, inputs):
        x, skip = inputs
        skip_flag = False
        for layer in self.layers:
            if skip_flag:
                x = x + skip
                skip_flag = False
            if isinstance(layer, torch.nn.UpsamplingBilinear2d):
                skip_flag = True
            
            x = layer(x)
            
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(torch.nn.Sequential(torch.nn.ReLU(),
                                               torch.nn.ReflectionPad2d(1),
                                               torch.nn.Conv2d(64, 3, kernel_size=3, padding=0)))
        
        self.blocks.append(DecoderBlock([
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=0)
        ]))

        self.blocks.append(DecoderBlock([
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=0)
        ]))

        #ADD LAST BLOCK
        self.blocks.append(DecoderBlock([
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=0)
        ]))

    def forward(self, block, x, skiplayer=None):
        if block != 0:
            x = [x, skiplayer]
        x = self.blocks[block](x)
        return x



def preprocess(img):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.CenterCrop((
        (img.height // 8) * 8,
        (img.width // 8) * 8))(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

