# import torch
# from torch import nn
# from torch.nn import functional as F
# from .resnet import resnet50_ls
# from torchvision.models.resnet import Bottleneck
# import random

# class TopBDNet(nn.Module):
#     def __init__(self, num_classes=0, neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, double_bottleneck=False, drop_bottleneck_features=False):
#         super(TopBDNet, self).__init__()
#         self.drop_bottleneck_features = drop_bottleneck_features
#         if neck:
#             self.bottleneck_global = nn.BatchNorm1d(512)
#             self.bottleneck_global.bias.requires_grad_(False)  # no shift
#             self.bottleneck_db = nn.BatchNorm1d(1024)
#             self.bottleneck_db.bias.requires_grad_(False)  # no shift
#             self.bottleneck_drop_bottleneck_features = nn.BatchNorm1d(2048)
#             self.bottleneck_drop_bottleneck_features.bias.requires_grad_(False)  # no shift
#         else:
#             self.bottleneck_global = None
#             self.bottleneck_db = None
#             self.bottleneck_drop_bottleneck_features = None

#         self.reduction_global = nn.Sequential(
#             nn.Conv2d(2048, 512, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         self.reduction_db = nn.Sequential(
#             nn.Linear(2048, 1024, 1),
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )

#         self.maxpool = nn.AdaptiveMaxPool2d((1,1))
#         self.avgpool_global = nn.AdaptiveAvgPool2d((1,1))
#         self.avgpool_drop = nn.AdaptiveAvgPool2d((1,1))
#         self.classifier_global = nn.Linear(512, num_classes)
#         self.classifier_db = nn.Linear(1024, num_classes)
#         self.batch_drop = BatchFeatureErase_Top(2048, drop_height_ratio, drop_width_ratio, double_bottleneck)
#         if self.drop_bottleneck_features:
#             self.classifier_drop_bottleneck = nn.Linear(2048, num_classes)
#         else:
#             self.classifier_drop_bottleneck = None
#         self._init_params()

#         resnet = resnet50_ls(num_classes, pretrained=True)
#         self.base = nn.Sequential(*list(resnet.children())[:-2])

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x, return_featuremaps = False, drop_top=False, visdrop=False):
#         base = self.base(x)

#         if visdrop: #return dropmask
#             drop_mask = self.batch_drop(base, drop_top=drop_top, visdrop=visdrop)
#             return drop_mask

#         if self.drop_bottleneck_features:
#             drop_x, t_drop_bottleneck_features = self.batch_drop(base, drop_top=drop_top, bottleneck_features = True)
#             t_drop_bottleneck_features = self.avgpool_drop(t_drop_bottleneck_features).view(t_drop_bottleneck_features.size()[:2])
#             if self.bottleneck_drop_bottleneck_features:
#                 x_drop_bottleneck_features = self.bottleneck_drop_bottleneck_features(t_drop_bottleneck_features)
#             else:
#                 x_drop_bottleneck_features = t_drop_bottleneck_features
#             x_drop_bottleneck_features = self.classifier_drop_bottleneck(x_drop_bottleneck_features)
#         else:
#             drop_x = self.batch_drop(base, drop_top=drop_top)

#         #global
#         x = self.avgpool_global(base)
#         t_x = self.reduction_global(x)
#         t_x = t_x.view(t_x.size()[:2])
#         if self.bottleneck_global:
#             x_x = self.bottleneck_global(t_x)
#         else:
#             x_x = t_x
#         x_prelogits = self.classifier_global(x_x)

#         #db
#         drop_x = self.maxpool(drop_x).view(drop_x.size()[:2])
#         t_drop_x = self.reduction_db(drop_x)
#         if self.bottleneck_db:
#             x_drop_x = self.bottleneck_db(t_drop_x)
#         else:
#             x_drop_x = t_drop_x
#         x_drop_prelogits = self.classifier_db(x_drop_x)

#         # if not self.training:
#         #     return x_prelogits, x_drop_prelogits

#         # if self.loss == 'triplet_dropbatch':
#         #     return x_prelogits, t_x, x_drop_prelogits, t_drop_x
#         # if self.loss == 'triplet_dropbatch_dropbotfeatures':
#         #     return x_prelogits, t_x, x_drop_prelogits, t_drop_x, x_drop_bottleneck_features, t_drop_bottleneck_features
#         # else:
#         #     raise KeyError("Unsupported loss: {}".format(self.loss))
#         if return_featuremaps:
#             return base, x_prelogits
#         # if return_featuremaps:
#         #     return base, x_prelogits, x_drop_prelogits, x_drop_bottleneck_features
#         # return x_prelogits, x_drop_prelogits
#         return x_prelogits, x_drop_bottleneck_features

#         # return x_prelogits

#     def name(self) -> str:
#         return 'inception_iccv'

# class BatchFeatureErase_Top(nn.Module):
#     def __init__(self, channels, h_ratio=0.33, w_ratio=1., double_bottleneck = False):
#         super(BatchFeatureErase_Top, self).__init__()
#         if double_bottleneck:
#             self.drop_batch_bottleneck = nn.Sequential(
#                 Bottleneck(channels, 512),
#                 Bottleneck(channels, 512)
#             )
#         else:
#             self.drop_batch_bottleneck = Bottleneck(channels, 512)

#         self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
#         self.drop_batch_drop_top = BatchDropTop(h_ratio)

#     def forward(self, x, drop_top=False, bottleneck_features = False, visdrop=False):
#         features = self.drop_batch_bottleneck(x)
#         if drop_top:
#             x = self.drop_batch_drop_top(features, visdrop=visdrop)
#         else:
#             x = self.drop_batch_drop_basic(features, visdrop=visdrop)
#         if visdrop:
#             return x #x is dropmask
#         if bottleneck_features:
#             return x, features
#         else:
#             return x

# class BatchDrop(nn.Module):
#     def __init__(self, h_ratio, w_ratio):
#         super(BatchDrop, self).__init__()
#         self.h_ratio = h_ratio
#         self.w_ratio = w_ratio
    
#     def forward(self, x, visdrop=False):
#         if self.training or visdrop:
#             h, w = x.size()[-2:]
#             rh = round(self.h_ratio * h)
#             rw = round(self.w_ratio * w)
#             sx = random.randint(0, h-rh)
#             sy = random.randint(0, w-rw)
#             mask = x.new_ones(x.size())
#             mask[:, :, sx:sx+rh, sy:sy+rw] = 0
#             if visdrop:
#                 return mask
#             x = x * mask
#         return x

# class BatchDropTop(nn.Module):
#     def __init__(self, h_ratio):
#         super(BatchDropTop, self).__init__()
#         self.h_ratio = h_ratio
    
#     def forward(self, x, visdrop=False):
#         if self.training or visdrop:
#             b, c, h, w = x.size()
#             rh = round(self.h_ratio * h)
#             act = (x**2).sum(1)
#             act = act.view(b, h*w)
#             act = F.normalize(act, p=2, dim=1)
#             act = act.view(b, h, w)
#             max_act, _ = act.max(2)
#             ind = torch.argsort(max_act, 1)
#             ind = ind[:, -rh:]
#             mask = []
#             for i in range(b):
#                 rmask = torch.ones(h)
#                 rmask[ind[i]] = 0
#                 mask.append(rmask.unsqueeze(0))
#             mask = torch.cat(mask)
#             mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
#             mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
#             if x.is_cuda: mask = mask.cuda()
#             if visdrop:
#                 return mask
#             x = x * mask
#         return x

# class BatchFeatureErase_Basic(nn.Module):
#     def __init__(self, channels, h_ratio=0.33, w_ratio=1.):
#         super(BatchFeatureErase_Basic, self).__init__()
#         self.drop_batch_bottleneck = Bottleneck(channels, 512)
#         self.drop_batch_drop = BatchDrop(h_ratio, w_ratio)

#     def forward(self, x):
#         x = self.drop_batch_bottleneck(x)
#         x = self.drop_batch_drop(x)
#         return x


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck
import random
import copy
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
from .resnet import resnet50_ls


class TopBDNet(nn.Module):
    def __init__(self, num_classes=0):
        super(TopBDNet, self).__init__()

        #global
        self.avgpool_global = nn.AdaptiveAvgPool2d((1,1))

        self.norm = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.bottleneck_global = nn.BatchNorm1d(512)
        self.bottleneck_global.bias.requires_grad_(False)  # no shift

        self.classifier_global = nn.Linear(512, num_classes)

        #k-means
        self.split = Kmeans_4p().cuda()
        self.classifier_kmeans = nn.Linear(512, num_classes)

        #base
        resnet = resnet50_ls(num_classes, pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        base = self.base(x)
        print(base.shape)

        #global
        x = self.avgpool_global(base)
        x = self.norm(x)
        x = x.view(x.size()[:2])
        x = self.bottleneck_global(x)
        x_prelogits = self.classifier_global(x)

        #kmeans
        y = self.split(base)
        y_prelogits = self.classifier_kmeans(y)



        return x_prelogits,y_prelogits

    def name(self) -> str:
        return 'inception_iccv'

# class Osnet(nn.Module):
#     def __init__(self):
#         super(Osnet, self).__init__()
#         osnet = osnet_x1_0(pretrained=True)
        
#         self.loss = loss
        
#         self.layer0 = nn.Sequential(
#             osnet.conv1,
#             osnet.maxpool
#             )
#         self.layer1 = osnet.conv2
#         self.layer2 = osnet.conv3
#         self.layer30 = osnet.conv4
#         self.layer40 = osnet.conv5 

#     def forward(self, x):
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x1 = self.layer30(x)
#         x1 = self.layer40(x1)
#         return x1


class Kmeans_4p(nn.Module):
    def __init__(self):
        super(Kmeans_4p, self).__init__()

        self.avgpool_kmeans1 = nn.AdaptiveAvgPool2d((1,1))
        self.norm_kmeans1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.bottleneck_kmeans1 = nn.BatchNorm1d(128)
        self.bottleneck_kmeans1.bias.requires_grad_(False)

        self.avgpool_kmeans2 = nn.AdaptiveAvgPool2d((1,1))
        self.norm_kmeans2 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.bottleneck_kmeans2 = nn.BatchNorm1d(128)
        self.bottleneck_kmeans2.bias.requires_grad_(False)

        self.avgpool_kmeans3 = nn.AdaptiveAvgPool2d((1,1))
        self.norm_kmeans3 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.bottleneck_kmeans3 = nn.BatchNorm1d(128)
        self.bottleneck_kmeans3.bias.requires_grad_(False)

        self.avgpool_kmeans4 = nn.AdaptiveAvgPool2d((1,1))
        self.norm_kmeans4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.bottleneck_kmeans4 = nn.BatchNorm1d(128)
        self.bottleneck_kmeans4.bias.requires_grad_(False)

    def __kmeans(self, x):

        # if torch.cuda.is_available():
        #     device = torch.device('cuda:0')
        # else:
        #     device = torch.device('cpu')


        # re1 = torch.zeros(x.size(), device=device)
        # re2 = torch.zeros(x.size(), device=device)
        # re3 = torch.zeros(x.size(), device=device)
        # re4 = torch.zeros(x.size(), device=device)

        re1 = torch.zeros(x.size())
        re2 = torch.zeros(x.size())
        re3 = torch.zeros(x.size())
        re4 = torch.zeros(x.size())

        

        for i in range(x.size()[0]):
            x1 = x[i]
            a, b, c = x1.shape
            x1=x1.reshape((a,b*c))
            # k means procedure
            cluster_ids_x, cluster_centers = kmeans(
                X=x1, num_clusters=4, distance='euclidean', tqdm_flag =False
                # , device=device
            )
            cluster_ids_y = kmeans_predict(
                x1, cluster_centers, 'euclidean', tqdm_flag = False
                # , device=device
            )
            re = []
            for k in range(4):
                temp = torch.zeros(x1.size()
                # , device=device
                )
                temp [k==cluster_ids_y] = x1[k==cluster_ids_y]
                temp = temp.reshape((a,b,c))
                re += [temp]

            re1[i] = re[0]
            re2[i] = re[1]
            re3[i] = re[2]
            re4[i] = re[3]

        return re1, re2, re3, re4

    def forward(self, x):
        x1 ,x2, x3, x4 = self.__kmeans(x)

        x1 = self.avgpool_kmeans1(x1)
        x1 = self.norm_kmeans1(x1)
        x1 = x1.view(x1.size()[:2])
        x1 = self.bottleneck_kmeans1(x1)

        x2 = self.avgpool_kmeans1(x2)
        x2 = self.norm_kmeans1(x2)
        x2 = x2.view(x2.size()[:2])
        x2 = self.bottleneck_kmeans2(x2)

        x3 = self.avgpool_kmeans1(x3)
        x3 = self.norm_kmeans1(x3)
        x3 = x3.view(x3.size()[:2])
        x3 = self.bottleneck_kmeans3(x3)

        x4 = self.avgpool_kmeans1(x4)
        x4 = self.norm_kmeans1(x4)
        x4 = x4.view(x4.size()[:2])
        x4 = self.bottleneck_kmeans4(x4)

        a = x4.shape[0]
        b = x4.shape[1]
        re = torch.zeros(a,b*4)
        for i in range(a):
            index = 0
            for j in [x1,x2,x3,x4]:
                for _ in range(b):
                    re[i][index] = j[i][_]
                    index +=1
        re = re
        return re
