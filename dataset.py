import os
from torch.utils.data import Dataset
import json
from PIL import Image
from util import Context, parse_args
import random
import numpy as np
from torchvision import transforms as transforms

args = parse_args()
Context().init_by_args(args)
config = Context().get_config()
logger = Context().get_logger()

DEBUG = Context().DEBUG

CROP_SIZE = config['transform']['patch_size']
CHANNEL_NUM = config['transform']['channel_num']

def isHorizontalFlip(p=0.5):
    if random.random() < p:
        return True
    else:
        return False

def randomCropper(img_size,patch_size=CROP_SIZE):
    h,w = img_size
    p_h,p_w = patch_size
    random_range_h =  h - p_h
    random_range_w = w - p_w
    def randomCrop(image_a,image_b):
        random_h=int(random.random()*random_range_h)
        random_w=int(random.random()*random_range_w)

        image_a=image_a[:,random_h:random_h+p_h,random_w:random_w+p_w]
        image_b=image_b[:,random_h:random_h+p_h,random_w:random_w+p_w]

        return image_a,image_b

    return  randomCrop


class ImageDataset(Dataset):


    def __init__(self, subj_score_file, directory , mode='train', channel=CHANNEL_NUM):

        if mode =='train':
            self.isTrain=True
        else:
            self.isTrain=False
        with open(subj_score_file, "r") as f:
            data = json.load(f)
        self.root_dir = directory
        data = data[mode]
        self.ref = data['ref']
        self.dis = data['dis']
        self.label = data['mos']
        self.channel = channel

        self.toTensor = transforms.Compose([
                    transforms.ToTensor(),
        ])

        self.horizontalFlip=transforms.Compose([
            transforms.RandomHorizontalFlip(1)
        ])

        self.norm=transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])


        augmentation_num=40

        if self.isTrain:
            self.dis=self.dis*augmentation_num
            self.ref=self.ref*augmentation_num
            self.label=self.label*augmentation_num
        else:
            dis=[]
            ref=[]
            label=[]
            for index in range(len(self.dis)):
                for times in range(augmentation_num):
                    dis.append(self.dis[index])
                    ref.append(self.ref[index])
                    label.append(self.label[index])
            self.dis=dis
            self.ref=ref
            self.label=label

    def __getitem__(self, index):
        label = self.label[index]
        ref = self.read_img(os.path.join(self.root_dir, self.ref[index]))
        dis = self.read_img(os.path.join(self.root_dir, self.dis[index]))

        if DEBUG:
            print('ref dimension after __getitem__: {}'.format(ref.shape))
            print('dis dimension after __getitem__: {}'.format(dis.shape))
            print('label value after __getitem__: {}'.format(label.numpy()))

        #flip
        if self.isTrain and isHorizontalFlip(0.5):
            ref=self.horizontalFlip(ref)
            dis=self.horizontalFlip(dis)
        #tensor
        ref=self.toTensor(ref)
        dis=self.toTensor(dis)
        #crop
        c,h,w=ref.shape
        randomCrop=randomCropper([h,w])
        ref,dis=randomCrop(ref,dis)
        #norm
        ref = self.norm(ref)
        dis = self.norm(dis)

        # [C, H,W]
        return ref, dis, label

    def read_img(self, img_path, resize=None):
        img = Image.open(img_path)
        if resize != None:
            img = img.resize(resize)
        if self.channel == 1:
            img = img.convert('L')

        #c,h,w
        return img

    def __len__(self):
        return len(self.dis)

