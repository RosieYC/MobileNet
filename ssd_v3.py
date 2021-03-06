import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from thop import profile


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
        self.mb = base #nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = extras#nn.ModuleList(extras)

        self.loc = head[0]#nn.ModuleList(head[0])
        self.conf = head[1]#nn.ModuleList(head[1])

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
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        '''
        tt = 0 
        k = 0 
        for layer in self.mb:
                    
          if k == 12:
            x = layer.conv[0](x)
            x = layer.conv[1](x)
            x = layer.conv[2](x)
            x = layer.conv[3](x)
            #print('lll: ' , layer.conv)
            #x = layer.conv[1][0](x)
            #print('loc2: ' , self.conf[tt](x).size()) # [8, 24, 19, 19]
            #print('conf2: ' , self.conf[tt](x).size()) # [8, 126, 19, 19]
            loc.append(self.loc[tt](x).permute(0,2,3,1).contiguous())
            conf.append(self.conf[tt](x).permute(0,2,3,1).contiguous())
            x = layer.conv[4](x)
            x = layer.conv[5](x)
            x = layer.conv[6](x)
            x = layer.conv[7](x)
            #x = layer.conv[1][1](x)
            #x = layer.conv[1][2](x)
            #x = layer.conv[2](x)
            #x = layer.conv[3](x)
            tt += 1 
          elif k == 18:
            x = layer[0](x)
            #print('loc2: ' , self.conf[tt](x).size())  # [8, 24, 10, 10]
            #print('conf2: ' , self.conf[tt](x).size()) # [8, 126, 10, 10]
            loc.append(self.loc[tt](x).permute(0,2,3,1).contiguous())
            conf.append(self.conf[tt](x).permute(0,2,3,1).contiguous())
            x = layer[1](x)
            x = layer[2](x)
            tt += 1 
          else:
            x = layer(x)
          k+=1 
          
          
        ### extras
        k = 0 
        for layer in self.extras:
          
          x = layer(x)
          #print('tt : ' , tt, x.size())
          #, self.loc[tt](x).size(), self.conf[tt](x).size())
          #print('loc: ' , self.conf[tt](x).size())
          
          loc.append(self.loc[tt](x).permute(0,2,3,1).contiguous())
          conf.append(self.conf[tt](x).permute(0,2,3,1).contiguous())
          tt += 1 
        #print('x_final_size: ' , x.size())
        ### 
        #x = [8, 512, 5, 5], loc[2](x) = [8, 24, 5, 5], conf[2](x) = [8, 126, 5, 5]
        #x = [8, 256, 3, 3], loc[3](x) = [8, 24, 3, 3], conf[3](x) = [8, 126, 3, 3]
        #x = [8, 256, 2, 2], loc[4](x) = [8, 24, 2, 2], conf[4](x) = [8, 126, 2, 2]
        #x = [8, 64, 1, 1],  loc[5](x) = [8, 24, 1, 1], conf[5](x) = [8, 126, 1, 1]
        
          
        
        ###
        '''
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            print('x: ' , x.size())
            print('l: ' , l)
            print('c: ' , c)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        '''
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #print('loc_size: ' , loc.size())
        #print('conf_size: ' , conf.size())
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

def multibox(vgg, extra_layers, cfg, num_classes):
    for i in range(len(vgg)):
      print(str(i) + ' : ' + str(vgg[i]))
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        print('xxxxxxxxxxxxxxxxxxxxxxxxx')
        
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        print('loc: ', loc_layers)
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
 
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
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
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
if __name__ == '__main__':
    net = build_ssd('test')
    model_path = r'/data/pro_SSD_pytorch/ssd.pytorch-master/weights/VOC.pth'
    model = net.load_state_dict(torch.load(model_path))
    input_ = torch.randn(1,3,300, 300)
    flops, params = profile(model, inputs=(input,), verbose=True)
    print("%s | %.2f | %.2f" % ('VOC', params / (1000 ** 2), flops / (1000 ** 3)))

