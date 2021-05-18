import torch
from torch import nn
import torch.nn.functional as F
from thop import profile 
from ptflops import get_model_complexity_info

def compute_FLOPs(model, input):
    flops, params = profile(model, inputs=(input, ))
    print(flos/1e9, params/1e6)
    # input = (3, 224, 224)
    flops, params = get_model_complexity_info(net, input, as_strings=True,print_per_layer_stat=True)
    print('flops: ' + flops)
    print('params: ' + params)



def _make_divisible(channel, divisor, min_channel=None):
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    # from original tf repo. 

    if min_channel is None:
        min_channel = divisor
    new_channel = max(min_channel, int(channel + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channel < 0.9 * channel:
        new_channel += divisor
    return int(new_channel)
def CONV_BN_RELU6(in_c, out_c, kernel_size, stride=1, groups=1, bias=False):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
    nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio_t):
        super(InvertedResidual, self).__init__()# python3.x
        # super(InvertedResidual, self).__init__() # python 2.x
        hidden_layer = in_ch * expand_ratio_t
        self.short_cut = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio_t == 1:
            layers.extend([
                # depth-wise 3x3
                CONV_BN_RELU6(hidden_layer, hidden_layer, 3, stride, groups=hidden_layer, bias=False),
                # nn.Conv2d(in_channels=hidden_layer, out_channels=hidden_layer, kernel_size=3, stride=stride, groups=hidden_layer, bias=False),
                # nn.BatchNorm2d(hidden_layer),
                # nn.ReLU6(inplace=True),
                # point-wise linear 1x1
                nn.Conv2d(hidden_layer, out_ch, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_ch),
                
                ])

        else:
            layers.extend([
                # point-wise 1x1 
                CONV_BN_RELU6(in_ch, hidden_layer, kernel_size=1,  bias=False),

                # nn.Conv2d(in_channels=in_ch, out_channels=hidden_layer,kernel_size=1, stride=1, groups=1,bias=False),
                # nn.BatchNorm2d(out_channels=hidden_layer),
                # nn.ReLU6(inplace=True),

                # depth-wise 3x3 
                CONV_BN_RELU6(hidden_layer, hidden_layer, 3, stride, groups=hidden_layer, bias=False),
                # nn.Conv2d(in_channels=hidden_layer, out_channels=hidden_layer, kernel_size=3, stride=stride, groups=hidden_layer, bias=False),
                # nn.BatchNorm2d(out_channels=hidden_layer),
                # nn.ReLU6(inplace=True),
                # point-wise linear 1x1 
                nn.Conv2d(hidden_layer, out_ch, kernel_size=1, stride=1, groups=1, bias=False),
                nn.BatchNorm2d(out_ch),

                ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.short_cut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilenetV2(nn.Module):
    def __init__(self, num_classes=100, alpha=1.0, divisor=8):
        super(MobilenetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32*alpha, divisor)
        last_channel = _make_divisible(1280*alpha, divisor)

        InvertedResidual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        layers =  []
        ## 224x224x3 == conv2d == t == - == c == 32 == n == 1 == s == 2 
        layers.append(CONV_BN_RELU6(3, input_channel, kernel_size=3, stride=2, groups=1))
        for t,c,n,s in InvertedResidual_setting:
            output_channel = _make_divisible(c*alpha, divisor)
            for i in range(n):
                stride = s if i == 0 else 1 
                layers.append(block(input_channel, output_channel, stride=stride, expand_ratio_t=t))
                input_channel = output_channel
        
        ### 7x7x320 == conv2d == t == - == c == 1280 == n == 1 == s ==1
        layers.append(CONV_BN_RELU6(input_channel, last_channel, kernel_size=1, stride=1))  
        print(layers)
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classf = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
            )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ## mean =0, variance = sqrt(2/ ((1 + a^2)*fan_in))
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                ## mean = 0, variance = gain * sqrt(2/(fan_in+fan_out)) # Glorot_initialisation
                # nn.init.xavier_normal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x ):
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1 )
        x = self.classf(x)
        return x 