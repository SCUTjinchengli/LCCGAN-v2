from __future__ import print_function
from opt import opt
import os
import time
import copy
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from model import _netG, _netD, _encoder, _decoder
import utils
import numpy as np


utils.seedSetting(opt)
utils.cudaSetting(opt)


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.netG = _netG(opt.anchor_num, opt.latent_dim, opt.nz, opt.ngf, opt.nc)
        self.netD = _netD(opt.nc, opt.ndf)
        self.encoder = _encoder(opt.nc, opt.ndf, opt.latent_dim)
        self.decoder = _decoder(opt.nc, opt.ngf, opt.latent_dim)
        self.learnBasis = nn.Linear(self.opt.anchor_num, self.opt.latent_dim, bias=False)
        self.learnCoeff = nn.Linear(self.opt.anchor_num, self.opt.batchSize_s2, bias=False)
        self.dataloader = torch.utils.data.DataLoader(utils.createDataSet(self.opt, self.opt.imageSize),
                                                      batch_size=self.opt.batchSize_s1,
                                                      shuffle=True, num_workers=int(self.opt.workers))

        self.criterion_bce = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss(reduction='elementwise_mean')
        self.criterion_l2 = nn.MSELoss(reduction='elementwise_mean')

        # initialize the optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.s3_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.s3_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerEncoder = optim.Adam(self.encoder.parameters(), lr=opt.s1_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerDecoder = optim.Adam(self.decoder.parameters(), lr=opt.s1_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerBasis = optim.Adam(self.learnBasis.parameters(), lr=opt.s2_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerCoeff = optim.Adam(self.learnCoeff.parameters(), lr=opt.s2_lr, betas=(opt.beta1, opt.beta2))

        # some variables
        input = torch.FloatTensor(opt.batchSize_s1, opt.nc, opt.imageSize, opt.imageSize)
        label = torch.FloatTensor(opt.batchSize_s3)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.one = self.one.cuda()
        self.mone = self.mone.cuda()
        if opt.cuda:
            input, label = input.cuda(), label.cuda()
            self.netD = utils.dataparallel(self.netD, opt.ngpu, opt.gpu)
            self.netG = utils.dataparallel(self.netG, opt.ngpu, opt.gpu)
            self.encoder = utils.dataparallel(self.encoder, opt.ngpu, opt.gpu)
            self.decoder = utils.dataparallel(self.decoder, opt.ngpu, opt.gpu)
            self.learnBasis = utils.dataparallel(self.learnBasis, opt.ngpu, opt.gpu)
            self.learnCoeff = utils.dataparallel(self.learnCoeff, opt.ngpu, opt.gpu)
            self.criterion_bce.cuda()
            self.criterion_l1.cuda()
            self.criterion_l2.cuda()

        self.input = Variable(input)
        self.label = Variable(label)
        self.batchSize = self.opt.batchSize_s1

    def cal_local_loss(self, recoverd, latent, basis, lcc_coding):
        batch_size = latent.size(0)
        latent_dim = latent.size(1)
        anchor_num = basis.size(0)
        assert (batch_size == lcc_coding.size(0))
        assert (latent_dim == basis.size(1))
        assert (anchor_num == lcc_coding.size(1))
        lcc_coding = lcc_coding / torch.sum(lcc_coding, dim=1, keepdim=True)
        # compute loss-1 and loss-3
        l1 = self.criterion_l2(recoverd, latent)
        l3 = self.criterion_l2(basis, torch.zeros_like(basis).cuda())
        # compute loss-2: local loss
        latent_expand = latent.view(batch_size, 1, latent_dim).expand(batch_size, anchor_num, latent_dim)
        basis_expand = basis.view(1, anchor_num, latent_dim).expand(batch_size, anchor_num, latent_dim)
        lcc_coding_expand = lcc_coding.abs()
        lcc_coding_expand = lcc_coding_expand.view(batch_size, anchor_num)
        v_minus_h = (torch.sum((latent_expand - basis_expand) ** 2, dim=2).sqrt()) ** 3
        l2 = torch.mean(lcc_coding_expand * v_minus_h)
        # see Appendix B for more details
        loss = 0.5 * self.opt.Lh * l1 + self.opt.Lv * l2 + self.opt.LCCLAMBDA * l3
        return loss

    def trainAutoEncoder(self):
        for epoch in range(self.opt.niter1):
            for i, data in enumerate(self.dataloader, 0):
                # gain the real data
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < opt.batchSize_s1:
                    break
                self.input.data.resize_(real_cpu.size()).copy_(real_cpu)
                self.encoder.zero_grad()
                self.decoder.zero_grad()

                input_hat = self.decoder(self.encoder(self.input))
                errRS = self.criterion_l1(input_hat, self.input)
                errRS.backward()
                self.optimizerEncoder.step()
                self.optimizerDecoder.step()
                print('[Stage1] [AutoEncoder] [epoch: %d/%d][batchSize: %d/%d] Loss_AE: %.4f' %
                      (epoch, self.opt.niter1, i, len(self.dataloader), errRS.item()))

    def trainLCC(self, s2_iters, s2_basis_iters, s2_coeff_iters):
        for epoch in range(self.opt.niter2):
            for i, data in enumerate(self.dataloader, 0):
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < self.opt.batchSize_s2:
                    break
                self.input.data.resize_(real_cpu.size()).copy_(real_cpu)
                latent = self.encoder(self.input).detach()
                prerec = self.decoder(latent)
                latent = latent.squeeze()
                # reset coeffients for new data
                self.learnCoeff.reset_parameters()

                for t in range(s2_iters):
                    ############################
                    # Stage2 (a) Train Coefficients
                    ############################
                    self.learnBasis.eval()
                    self.learnCoeff.train()
                    # basis_T: latent_dim x anchor_num
                    basis_T = self.learnBasis.weight.detach()
                    for j in range(s2_coeff_iters):
                        self.learnCoeff.zero_grad()
                        output = self.learnCoeff(basis_T).transpose(0, 1).contiguous()
                        # LCC Coding: batch_size x anchor_num
                        lcc_coding = self.learnCoeff.weight
                        loss_coeff = self.cal_local_loss(output, latent, basis_T.transpose(0, 1).contiguous(), lcc_coding)
                        loss_coeff.backward()
                        self.optimizerCoeff.step()
                        print('[Stage2] [Coeff] [epoch: %d/%d][batchSize: %d/%d]'
                              '[s2_iters: %d/%d] loss_coeff: %.4f learnCoeff_weights: %.4f' %
                              (epoch, self.opt.niter2, i, len(self.dataloader), t, s2_iters,
                               loss_coeff.data[0], torch.mean(self.learnCoeff.weight).item()))

                    ############################
                    # Stage2 (b) Learn Basis
                    ############################
                    self.learnBasis.train()
                    self.learnCoeff.eval()
                    # LCC Coding: batch_size x anchor_num
                    lcc_coding = self.learnCoeff.weight.detach()
                    for j in range(s2_basis_iters):
                        self.learnBasis.zero_grad()
                        output = self.learnBasis(lcc_coding)
                        # basis: anchor_num x latent_dim
                        basis = self.learnBasis.weight.transpose(0, 1).contiguous()
                        loss_basis = self.cal_local_loss(output, latent, basis, lcc_coding)
                        loss_basis.backward()
                        self.optimizerBasis.step()
                        print('[Stage2] [Basis] [epoch: %d/%d][batchSize: %d/%d]'
                              '[s2_iters: %d/%d] loss_basis: %.4f learnBasis_weights: %.4f' %
                              (epoch, self.opt.niter2, i, len(self.dataloader), t, s2_iters,
                               loss_basis.data[0], torch.mean(self.learnBasis.weight).item()))

    def trainGAN(self):
        for epoch in range(self.opt.niter3):
            for i, data in enumerate(self.dataloader, 0):
                self.netG.train()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < self.opt.batchSize_s3:
                    break
                self.input.data.resize_(real_cpu.size()).copy_(real_cpu)
                self.label.data.resize_(batch_size).fill_(1)
                ############################
                # (1) Update D network
                ############################
                self.netD.zero_grad()
                # train with real
                output = self.netD(self.input)
                errD_real = self.criterion_bce(output, self.label)
                errD_real.backward()
                # train with fake
                noise = torch.randn(batch_size, self.opt.nz)
                noise = noise.cuda()
                noisev = autograd.Variable(noise)
                fake = self.netG(noisev)

                self.label.data.resize_(batch_size).fill_(0)
                output = self.netD(fake.detach())

                errD_fake = self.criterion_bce(output, self.label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                self.optimizerD.step()
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################
                self.netG.zero_grad()
                self.label.data.fill_(1)  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = self.criterion_bce(output, self.label)
                errG.backward()
                D_G_z2 = output.data.mean()
                self.optimizerG.step()
                print('[Stage3] [GAN] [epoch: %d/%d][batchSize: %d/%d] errD: %.4f, errG: %.4f' %
                      (epoch, self.opt.niter3, i, len(self.dataloader), errD.item(), errG.item()))
                
    def train(self):
        start_time = time.time()
        end_time = start_time
        self.real_label = 1
        self.fake_label = 0
        ############################
        # Stage1: Autoencoder
        ############################
        self.trainAutoEncoder()

        ############################
        # Stage2: Train LCC
        ############################
        s2_iters = 10
        s2_basis_iters = 10
        s2_coeff_iters = 10
        self.dataloader = torch.utils.data.DataLoader(utils.createDataSet(self.opt, self.opt.imageSize),
                                                      batch_size=self.opt.batchSize_s2,
                                                      shuffle=True, num_workers=int(self.opt.workers))
        self.trainLCC(s2_iters, s2_basis_iters, s2_coeff_iters)

        ############################
        # Stage3: Training GAN
        ############################
        self.netG.reset_basis(self.learnBasis.weight.transpose(0, 1).contiguous())
        self.dataloader = torch.utils.data.DataLoader(utils.createDataSet(self.opt, self.opt.imageSize),
                                                      batch_size=self.opt.batchSize_s3 * self.opt.criticIters,
                                                      shuffle=True, num_workers=int(self.opt.workers))
        self.trainGAN()


if __name__ == '__main__':
    trainer = Trainer(opt)
    trainer.train()