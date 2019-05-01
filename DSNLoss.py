import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from layers.modules import MultiBoxLoss
from pyramidbox import Flatten
from data.config import cfg, args

flatten = Flatten()
img_to_tensor = ToTensor()

class DSNLoss(nn.Module):
    def __init__(self, ALPHA=cfg.ALPHA, BETA=cfg.BETA, GAMMA=cfg.GAMMA):
        super(DSNLoss, self).__init__()

        # DSN hyperparameters
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

        self.criterion1 = MultiBoxLoss(cfg, args.cuda)
        self.criterion2 = MultiBoxLoss(cfg, args.cuda, use_head_loss=True)
        self.reconstructLoss = nn.MSELoss()
        self.BCE = nn.BCELoss()

    def forward(self,
                output_detect, face_targets, head_targets,  # Detection results
                imgs_t, imgs_s, imgs_t_recon, imgs_s_recon,  # Reconst vs ori
                h_t, h_s, h_t_share, h_s_share,  # Feat diff loss
                domain_predict_t, domain_predict_s  # Domain classifier results
                ):
        # Detection Loss
        face_loss_l, face_loss_c = self.criterion1(output_detect, face_targets)
        head_loss_l, head_loss_c = self.criterion2(output_detect, head_targets)
        loss_detect = face_loss_l + face_loss_c + head_loss_l + head_loss_c
        # Reconstruction Loss
        #XXX
        # print(torch.max(imgs_t[0]))
        # print(torch.max(imgs_t_recon[0]))
        imgs_t = imgs_t / 255
        imgs_s = imgs_s / 255
        loss_recon_t = self.reconstructLoss(imgs_t, imgs_t_recon)
        loss_recon_s = self.reconstructLoss(imgs_s, imgs_s_recon)
        loss_recon = loss_recon_t + loss_recon_s
        # Embedding feature different loss
        loss_diff_s = torch.norm(torch.mm(flatten(h_s_share).t(), flatten(h_s)))
        loss_diff_t = torch.norm(torch.mm(flatten(h_t_share).t(), flatten(h_t)))
        loss_diff = loss_diff_t + loss_diff_s
        # Embedding feature similarity loss
        # Source: 0; Target: 1
        t_gt_labels = torch.ones(domain_predict_t.size())
        s_gt_labels = torch.zeros(domain_predict_s.size())

        loss_sim = -1 * (self.BCE(domain_predict_t, t_gt_labels) + self.BCE(domain_predict_s, s_gt_labels))
        # TODO
        # print('>>> In DSNloss')
        # print(loss_detect.item())
        # print(loss_recon.item())
        # print(loss_diff.item())
        # print(loss_sim.item())
        loss = loss_detect + self.ALPHA * loss_recon + \
            self.BETA * loss_diff + self.GAMMA * loss_sim
        return loss, face_loss_l, face_loss_c, head_loss_l, head_loss_c

