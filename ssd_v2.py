import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from thop import profile
from mobilenetv2 import *

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.mb = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        '''
        # apply vgg up to conv4_3 relu
        for k in range(99):
            x = self.mb[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.mb[k](x)
        sources.append(x)
        '''
        old = 1
        for k in range(len(self.mb)):
          #print('k: ' , k )
          #print(self.mb[k])
          if k >= 15 and (k+1)%8 ==0 :
            old_output = self.mb[k](x) 
            x = old_output
          if k>= 15 and k%8 == 0 :
            x += old_output
            
            x = self.mb[k](x)
            #print('x_size: ' , self.mb[k], k)
            if k in [32, 56, 72, 80, 96, 112]:
              sources.append(x)
            '''
            if old == 1:
              print(old)
              old_output = self.mb[k](x)
              x = old_output
              print(old_output.size())
              old = 0 
            else:
              print('aa: ', x.size())
              print('qq: ' , old_output.size())
              print('rrr: ' , x + old_output)
              x += old_output
              '''
          else:
            #print('ere')
            x = self.mb[k](x)
         
        #print('xxx: ', x.size())   
        #print('start extra layers')
        # apply extra layers and cache source layer outputs
        #print(self.extras)
        #for k, v in enumerate(self.extras):
            
        #    x = v(x)
        #    #print('here')
        #    if k % 2 == 1:
        #        #print(self.extras[k])
        #        #print(x.size())
        #        sources.append(x)
        #print('start apply multibox')
        # apply multibox head to source layers
        #print('loc: ' , self.loc)
        #print('conf:  ' , self.conf)
        #print('sources: ' , sources)
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #print('wwwwwwwwwwwwwwwwwwwwww')
            #print('x: ' , x.size())
            #print('l: ' , l)
            #print('c: ' , c)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    #print(layers)
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    #print(layers)
    return layers
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
def mobilenet_arch(alpha=1.0, round_nearest=8):
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    def CBNR(in_channel, out_channel, kernel_size=3, stride=1, groups=1):
      padding = (kernel_size - 1) // 2
      return [nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)]
            
    def IDRS(in_channel, out_channel, stride, expand_ratio):
      hidden_channel = in_channel * expand_ratio
      use_shortcut = stride == 1 and in_channel == out_channel

      layers = []
      if expand_ratio != 1:
          # 1x1 pointwise conv
          layers += CBNR(in_channel, hidden_channel, kernel_size=1)
      layers += CBNR(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel)# 3x3 depthwise conv
          # 1x1 pointwise conv(linear)
      layers += [nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
          nn.BatchNorm2d(out_channel)
      ]
      return layers 
      
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    features = []
    # conv1 layer
    #features.append(ConvBNReLU(3, input_channel, stride=2))
    features = CBNR(3, input_channel, stride=2)
    
    # building inverted residual residual blockes
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            features += IDRS(input_channel, output_channel, stride, expand_ratio=t)
            input_channel = output_channel
    # building last several layers
    features += CBNR(input_channel, last_channel, 1)
    #print(features)
    return features
def multibox(mb, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mb_source = [32, 56, 72, 80, 96, 112]
    #print('here: ',len(mb))
    #for i in range(len(mb)):
    #  print(str(i) + ' : ' + str(mb[i]))
    for k, v in enumerate(mb_source):
        loc_layers += [nn.Conv2d(mb[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mb[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    '''
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    '''
    #print('loc_layers: ' , loc_layers)
    #print('conf_layers: ' , conf_layers)
    return mb, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],#, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(mobilenet_arch(),
                                     add_extras(extras[str(size)], 1280),
                                     mbox[str(size)], num_classes)
    print('based_: ' , base_)
    print('extras_ : ', extras_)
    print('head_:', head_)
    return SSD(phase, size, base_, extras_, head_, num_classes)
if __name__ == '__main__':
    net = build_ssd('test')
    model_path = r'/data/pro_SSD_pytorch/ssd.pytorch-master/weights/VOC.pth'
    model = net.load_state_dict(torch.load(model_path))
    input_ = torch.randn(1,3,300, 300)
    flops, params = profile(model, inputs=(input,), verbose=True)
    print("%s | %.2f | %.2f" % ('VOC', params / (1000 ** 2), flops / (1000 ** 3)))

