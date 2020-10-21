from util import Context
from basic_block import *
from pytorch_ssim import SSIM
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo


config = Context().get_config()
logger = Context().get_logger()
# model
CROP_SIZE = config['transform']['patch_size']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_1 = self.layer1(x)

        x = self.layer2(x_1)
        x = self.layer3(x)
        x = self.layer4(x)



        return x_1,x


def resnet50_backbone( pretrained=False, **kwargs):


    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model



def weights_init_xavier(m):
    classname = m.__class__.__name__


    if classname.find('ConcatConv') != -1:
        init.kaiming_normal_(m.conv.weight.data)
    elif classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



class PredictNet(nn.Module):
    def __init__(self, input_channel):
        super(PredictNet, self).__init__()
        self.input_channel=input_channel

        self.relu = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.downsample_filter = DownSampleFilter()
        self.upsample_filter = UpSampleFilter()

        self.mse_conv1 = ConcatConv(256+1+64, 64, kernel_size=3, stride=1,padding=1)
        self.mse_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.mse_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mse_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mse_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mse_conv6 = nn.Conv2d(64, 1,kernel_size=1, stride=1, padding=(0, 0))



        self.ssim_conv1 = ConcatConv(192, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.ssim_conv2 = ConcatConv(128, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.ssim_conv3 = ConcatConv(128, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.ssim_conv4 = ConcatConv(128, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.ssim_conv5 = ConcatConv(64+1, 1, kernel_size=1, stride=1, padding=(0, 0))

        self.predict_conv1 = ConcatConv(192, 64, kernel_size=3, stride=2, padding=(1, 1))
        self.predict_conv2 = ConcatConv(192, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.predict_conv3 = ConcatConv(192, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.predict_conv4 = nn.Conv2d(64, 1*2, kernel_size=1, stride=1)


        self.h1=  OneTrans(64,64)
        self.h2 = OneTrans(64,64)
        self.h3 = OneTrans(64,64)
        self.h4 = OneTrans(64,64)
        self.h5 = OneTrans(64,64)

        self.mse_net = [self.mse_conv1, self.mse_conv2, self.mse_conv3, self.mse_conv4,self.mse_conv5,self.mse_conv6]
        self.ssim_net=[self.ssim_conv1,self.ssim_conv2,self.ssim_conv3,self.ssim_conv4,self.ssim_conv5]
        self.predict_net=[self.predict_conv1,self.predict_conv2,self.predict_conv3,self.predict_conv4]
        self.hyper=[self.h1,self.h2,self.h3,self.h4,self.h5]

    def forward(self, x,high_level_feature):
        b,_,__,___=x.shape

        result_list=[]
        out=x
        #mse
        h0=self.hyper[0](high_level_feature)
        h0=self.upsample_filter(h0)
        out = self.mse_net[0](out,h0)
        out = self.relu(out)
        result_list.append(out)
        for index in  range(1,len(self.mse_net)-1):
            out=self.mse_net[index](out)
            out=self.relu(out)
            result_list.append(out)
        out=self.mse_net[-1](out)
        out=self.relu2(out)
        result_list.append(out)
        mse_error_map=out

        #ssim
        h1=self.hyper[1](high_level_feature)
        h1=self.upsample_filter(h1)
        out = self.ssim_net[0](result_list[0],[result_list[1],h1])
        out = self.relu(out)
        result_list[1]=out
        for index in  range(1,len(self.ssim_net)-1):
            out=self.ssim_net[index](result_list[index],result_list[index+1])
            out=self.relu(out)
            result_list[index+1]=out
        out = self.ssim_net[-1](out,mse_error_map)
        out=self.relu2(out)
        ssim_error_map=out

        #predict mos
        out=x
        for index in range(len(self.predict_net)-1):
            feature_1 = result_list[index+1]
            feature_2 = result_list[index+2]
            feature_3=self.hyper[index+2](high_level_feature)
            if index==0:
                feature_3=self.upsample_filter(feature_3)

            if index>=1:
                feature_2=self.downsample_filter(feature_2)

            out = self.predict_net[index](feature_1,[feature_2,feature_3])
            out = self.relu(out)
            result_list[index+2]=out
        out=self.predict_net[-1](out)
        sense_map=self.relu2(out)


        return sense_map,mse_error_map,ssim_error_map


class OneTrans(nn.Module):
    def __init__(self,in_,out_):
        super(OneTrans, self).__init__()
        self.conv=nn.Conv2d(in_,out_,kernel_size=1,stride=1,padding=0)
        self.relu=nn.LeakyReLU(inplace=True)

    def forward(self, x,times=4):
        x=self.conv(x)
        x=self.relu(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()


        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse= nn.MSELoss(reduce=True, size_average=True)
        self.l1=nn.L1Loss()
        self.SL1=nn.SmoothL1Loss()
        self.ssim=SSIM(window_size = 11)
        self.avg=nn.AdaptiveAvgPool2d(1)


        self.predict_net = PredictNet(64)
        self.device = next(self.predict_net.parameters()).device


        self.resnet = resnet50_backbone(pretrained=True)
        self.regression=nn.Sequential(
            nn.Conv2d(1*2, 1, kernel_size=1, stride=1,padding=0),
            nn.LeakyReLU(inplace=True),
        )

        self.res_out=nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1,padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )


        init_nets=[self.regression,self.predict_net,self.res_out]
        for net in init_nets:
            net.apply(weights_init_xavier)



        def get_log_diff_fn(eps=0.2):
            log_255_sq = np.float32(2 * np.log(255.0))
            log_255_sq = log_255_sq.item()  # int
            max_val = np.float32(log_255_sq - np.log(eps))
            max_val = max_val.item()  # int
            log_255_sq = torch.from_numpy(np.array(log_255_sq)).float().to(self.device)
            max_val = torch.from_numpy(np.array(max_val)).float().to(self.device)

            def log_diff_fn(in_a, in_b):
                diff = 255.0 * (in_a - in_b)
                val = log_255_sq - torch.log(diff ** 2 + eps)
                return val / max_val

            return log_diff_fn

        self.log_diff_fn = get_log_diff_fn(1)


        self.downsample_filter = DownSampleFilter()
        self.upsample_filter = UpSampleFilter()

    def rgb2gray(self,rgb):
        p,c,w,h=rgb.shape
        r, g, b = rgb[:, 0, :,:], rgb[:, 1, :,:], rgb[:, 2, :,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray.reshape(p,1,w,h)

    def forward(self, r_patch_set, d_patch_set, mos_set):
        error_map = self.log_diff_fn(self.rgb2gray(r_patch_set), self.rgb2gray(d_patch_set))
        eds_4=self.downsample_filter(self.downsample_filter(error_map))

        d_g=self.rgb2gray(d_patch_set)
        d_g_4=self.downsample_filter(self.downsample_filter(d_g))
        l_d_g_4=self.lowpass(d_g_4,3)
        h_d_g_4=d_g_4-l_d_g_4


        x_1,x = self.resnet(d_patch_set)
        x_1=torch.cat([x_1,h_d_g_4],dim=1)
        x=self.res_out(x)
        x=self.upsample_filter(x)
        sense_map, low_img, generate_img=self.predict_net(x_1,x)


        ge_i=torch.cat([self.downsample_filter(generate_img),self.downsample_filter(low_img)],dim=1)
        out=sense_map*ge_i
        out=shave_border(out)
        out=self.regression(out)
        mos=torch.mean(out,dim=(1,2,3))



        mos = mos.reshape(mos_set.shape).float()
        mos_set = mos_set.float()

        subj_loss = self.SL1(mos, mos_set)
        error_loss = self.mse(eds_4, low_img)
        ssim_loss=1-self.ssim(eds_4,generate_img)


        total_loss=subj_loss*1e3+error_loss*5e1+ssim_loss*15e1

        return total_loss,mos




    def lowpass(self, img, n_level):
        '''Normalize image by subtracting the low-pass-filtered image'''
        # Downsample
        img_ = img
        pyr_sh = []
        for i in range(n_level - 1):
            pyr_sh.append(img_.shape)
            img_ = self.downsample_filter(img_)
        # Upsample
        for i in range(n_level - 1):
            img_ = self.upsample_filter(img_)
        return img_




