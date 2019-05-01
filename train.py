#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data

import os
import time
import torch

import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg, args
from pyramidbox import build_net
from data.dataset import DSNDataset, detection_collate
from DSNLoss import DSNLoss

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

train_dataset = DSNDataset(cfg.FACE.TRAIN_FILE_S, cfg.FACE.TRAIN_FILE_T, mode='train')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_dataset = DSNDataset(cfg.FACE.VAL_FILE_S, cfg.FACE.VAL_FILE_T, mode='val')
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
criterion = DSNLoss()

def train():
    iteration = 0
    start_epoch = 0
    step_index = 0
    per_epoch_size = len(train_dataset) // args.batch_size

    pyramidbox = build_net('train', cfg.NUM_CLASSES)
    net = pyramidbox
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        # vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Load base network....')
        # XXX: rename the keys
        # net.vgg.load_state_dict(vgg_weights, strict=False)
        # net.load_state_dict(vgg_weights, strict=False)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(pyramidbox)
        net = net.cuda()
        cudnn.benckmark = True

    if not args.resume:
        def print_model(model):
            for param in model.parameters():
                print(param.data)
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        print('Initializing weights...')
        net.apply(weights_init)
        # pyramidbox.extras.apply(pyramidbox.weights_init)
        # pyramidbox.lfpn_topdown.apply(pyramidbox.weights_init)
        # pyramidbox.lfpn_later.apply(pyramidbox.weights_init)
        # pyramidbox.cpm.apply(pyramidbox.weights_init)
        # pyramidbox.loc_layers.apply(pyramidbox.weights_init)
        # pyramidbox.conf_layers.apply(pyramidbox.weights_init)


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)
    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (imgs_s, face_targets_s, head_targets_s,
                imgs_t, face_targets_t, head_targets_t) in enumerate(train_loader):
            if args.cuda:
                imgs_s = Variable(imgs_s.cuda())
                imgs_t = Variable(imgs_t.cuda())
                face_targets_s = [Variable(ann.cuda(), volatile=True)
                                for ann in face_targets_s]
                head_targets_s = [Variable(ann.cuda(), volatile=True)
                                for ann in head_targets_s]
                face_targets_t = [Variable(ann.cuda(), volatile=True)
                                for ann in face_targets_t]
                head_targets_t = [Variable(ann.cuda(), volatile=True)
                                for ann in head_targets_t]
            else:
                imgs_s = Variable(imgs_s)
                imgs_t = Variable(imgs_t)
                face_targets_s = [Variable(ann, volatile=True)
                                for ann in face_targets_s]
                head_targets_s = [Variable(ann, volatile=True)
                                for ann in head_targets_s]
                face_targets_t = [Variable(ann, volatile=True)
                                for ann in face_targets_t]
                head_targets_t = [Variable(ann, volatile=True)
                                for ann in head_targets_t]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            # TODO
            # imgs_t = images
            # imgs_s = images
            out = net(imgs_t, imgs_s)
            output_detect, imgs_t_recon, imgs_s_recon, \
                h_t, h_t_share, h_s, h_s_share, \
                domain_predict_t, domain_predict_s = out
            # backprop
            optimizer.zero_grad()
            # TODO import DSNLoss
            loss, face_loss_l, face_loss_c, head_loss_l, head_loss_c =\
                criterion(
                    output_detect, face_targets_s, head_targets_s,
                    imgs_t, imgs_s, imgs_t_recon, imgs_s_recon,
                    h_t, h_s, h_t_share, h_s_share,
                    domain_predict_t, domain_predict_s
                    )
            losses += loss.data[0]
            loss.backward()
            optimizer.step()
            t1 = time.time()
            face_loss = (face_loss_l + face_loss_c).data[0]
            head_loss = (head_loss_l + head_loss_c).data[0]

            if iteration % 10 == 0:
                loss_ = losses / (batch_idx + 1)
                print('Timer: {:.4f} sec.'.format(t1 - t0))
                print('epoch ' + repr(epoch) + ' iter ' +
                      repr(iteration) + ' || Loss:%.4f' % (loss_))
                print('->> face Loss: {:.4f} || head loss : {:.4f}'.format(
                    face_loss, head_loss))
                print('->> lr: {}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'pyramidbox_' + repr(iteration) + '.pth'
                torch.save(pyramidbox.state_dict(),
                           os.path.join(args.save_folder, file))
            iteration += 1

        val(epoch, net, pyramidbox)


def val(epoch,
        net,
        pyramidbox
        ):
    print(">>> In val:")
    net.eval()
    face_losses = 0
    head_losses = 0
    step = 0
    t1 = time.time()
    for batch_idx, (imgs_s, face_targets_s, head_targets_s,
                    imgs_t, face_targets_t, head_targets_t) \
            in enumerate(val_loader):
        if args.cuda:
            imgs_s = Variable(imgs_s.cuda())
            imgs_t = Variable(imgs_t.cuda())
            face_targets_s = [Variable(ann.cuda(), volatile=True)
                            for ann in face_targets_s]
            head_targets_s = [Variable(ann.cuda(), volatile=True)
                            for ann in head_targets_s]
            face_targets_t = [Variable(ann.cuda(), volatile=True)
                            for ann in face_targets_t]
            head_targets_t = [Variable(ann.cuda(), volatile=True)
                            for ann in head_targets_t]
        else:
            imgs_s = Variable(imgs_s)
            imgs_t = Variable(imgs_t)
            face_targets_s = [Variable(ann, volatile=True)
                            for ann in face_targets_s]
            head_targets_s = [Variable(ann, volatile=True)
                            for ann in head_targets_s]
            face_targets_t = [Variable(ann, volatile=True)
                            for ann in face_targets_t]
            head_targets_t = [Variable(ann, volatile=True)
                            for ann in head_targets_t]

        # out = net(imgs_s)
        # face_loss_l, face_loss_c = criterion1(out, face_targets_s)
        # head_loss_l, head_loss_c = criterion2(out, head_targets_s)

        out = net(imgs_t, imgs_s)
        output_detect, imgs_t_recon, imgs_s_recon, \
            h_t, h_t_share, h_s, h_s_share, \
            domain_predict_t, domain_predict_s = out
        # TODO import DSNLoss
        loss, face_loss_l, face_loss_c, head_loss_l, head_loss_c =\
            criterion(
                output_detect, face_targets_s, head_targets_s,
                imgs_t, imgs_s, imgs_t_recon, imgs_s_recon,
                h_t, h_s, h_t_share, h_s_share,
                domain_predict_t, domain_predict_s
                )

        face_losses += (face_loss_l + face_loss_c).data[0]
        head_losses += (head_loss_l + head_loss_c).data[0]
        step += 1

    tloss = face_losses / step

    t2 = time.time()
    print('test Timer:{:.4f} .sec'.format(t2 - t1))
    print('epoch ' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(pyramidbox.state_dict(), os.path.join(
            args.save_folder, 'pyramidbox.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': pyramidbox.state_dict(),
    }
    torch.save(states, os.path.join(
        args.save_folder, 'pyramidbox_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
