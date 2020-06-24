# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/20 9:40
filename(文件名): train.py
function description(功能描述):
    OGNet的训练
"""
from network import g_net, d_net
from dataloader import load_data
from opts import parse_opts
Opt = parse_opts()
Opt.data_path = './data/train/'
data_loader = load_data(Opt)
from torch.optim import Adam
from torchvision.utils import save_image
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
plt.ion()


def unfreeze_model(Model):
    """
    unfreeze the model
    :param Model:
    :return:
    """
    for param in Model.parameters():
        param.requires_grad_(True)


def freeze_model(Model):
    """
    freeze the model
    :param Model:
    :return:
    """
    for param in Model.parameters():
        param.requires_grad_(False)


class G_pseudo(nn.Module):
    def __init__(self, Generator_old, Generator):
        super(G_pseudo, self).__init__()
        self.G_old = Generator_old
        self.G = Generator

    def forward(self, Input1, Input2):
        X_head = torch.div(torch.add(self.G_old(Input1), self.G_old(Input2)), 2)
        return self.G(X_head)


def denoising_image(Image: torch.Tensor, denoising_factor=0.5):
    """
    add noise for the input image.
    :param Image:
    :param denoising_factor:
    :return:
    """
    return torch.add(Image, torch.mul(torch.randn_like(Image), denoising_factor))


def L2Loss(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate the L2Loss of two tensors with has the same size.
    :param tensor1:
    :param tensor2:
    :return:
    """
    assert tensor1.shape == tensor2.shape
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    return torch.pow(torch.mean(torch.pow(torch.sub(tensor1, tensor2), 2)), 0.5)


def L_G_D(Input: torch.Tensor, Generator, Discriminator):
    """
    calculate loss of discriminator.
    :param Input:
    :param Generator:
    :param Discriminator:
    :return:
    """
    reconstruct_image = Generator(denoising_image(Input, 0.05))
    # print('HI')

    # training the discriminator
    discriminator_score1 = Discriminator(reconstruct_image)
    discriminator_score2 = Discriminator(Input)
    loss_func = nn.BCEWithLogitsLoss()
    loss1 = loss_func(discriminator_score1, torch.zeros_like(discriminator_score1))
    loss2 = loss_func(discriminator_score2, torch.ones_like(discriminator_score2))
    discriminator_loss = torch.add(loss1, loss2)
    return discriminator_loss, reconstruct_image


def train_phase_1(opt):
    print('training on ', opt.device)
    print("Training phase 1...")
    print("The goal is to obtain a generator that can well reconstruct the input.")
    generator = g_net().to(opt.device)
    old_generator = None
    discriminator = d_net().to(opt.device)
    generator_optimizer = Adam(generator.parameters(), lr=Opt.glr)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=Opt.dlr)

    for epoch in range(opt.phase1_epochs):
        print('-----------------------epoch {}--------------------------'.format(epoch))
        # num = 0
        for image, label in tqdm(data_loader):
            # denoised_image = denoising_image(image, 0.05)
            # reconstruct_image = generator(denoised_image)
            # # print('HI')
            #
            # # training the discriminator
            # discriminator_score1 = discriminator(reconstruct_image)
            # discriminator_score2 = discriminator(image)
            # loss_func = nn.BCEWithLogitsLoss()
            # loss1 = loss_func(discriminator_score1, torch.zeros_like(discriminator_score1))
            # loss2 = loss_func(discriminator_score2, torch.ones_like(discriminator_score2))
            # discriminator_loss = torch.add(loss1, loss2)
            image = image.to(opt.device)
            # discriminator_loss, reconstruct_image = L_G_D(image, generator, discriminator)
            reconstruct_image = generator(denoising_image(image, 0.05))

            # discriminator loss
            discriminator_score1 = discriminator(reconstruct_image)
            discriminator_score2 = discriminator(image)
            loss_func = nn.BCEWithLogitsLoss()
            loss1 = loss_func(discriminator_score1, torch.zeros_like(discriminator_score1))
            loss2 = loss_func(discriminator_score2, torch.ones_like(discriminator_score2))
            discriminator_loss = torch.add(loss1, loss2)

            discriminator_score_ = discriminator(reconstruct_image)
            generator_loss = loss_func(discriminator_score_, torch.ones_like(discriminator_score_))
            generator_loss = generator_loss + opt.lamb * L2Loss(image, reconstruct_image)

            # update discriminator
            freeze_model(generator)
            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            unfreeze_model(generator)

            # update generator
            freeze_model(discriminator)
            generator.zero_grad()
            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()
            unfreeze_model(discriminator)
        #     display
        #     plt.subplot(131), plt.imshow(image.detach().numpy()[0][0], cmap='gray')
        #     image_denoise = denoising_image(image, 0.05)
        #     plt.subplot(132), plt.imshow(image_denoise.detach().numpy()[0][0], cmap='gray')
        #     plt.subplot(133), plt.imshow(reconstruct_image.detach().numpy()[0][0], cmap='gray')
        #     plt.pause(0.5)
        # plt.ioff()
        # plt.show()

        if epoch == opt.g_old_epoch:
            torch.save(generator, './models/old_generator.pkl')
            old_generator = generator
    print("phase 1 training finished and now saving generator...")
    torch.save(generator, './models/trained_genenrator.pkl')
    print("trained generator saved!!!")
    print("saving discriminator for real and fake...")
    torch.save(discriminator, './models/discriminator_real_fake.pkl')
    print("discriminator for real and fake saved!!!")
    return old_generator, generator, discriminator


def train_phase_2(opt, old_generator, generator, discriminator):
    old_generator = old_generator.to(opt.device)
    generator = generator.to(opt.device)
    discriminator = discriminator.to(opt.device)
    print("Training phase 2...")
    print("The goal is to obtain a discriminator that can well distinguish good reconstruction and bad.")
    discriminator_optimizer = Adam(discriminator.parameters(), lr=opt.dlr)
    pseudo_generator = G_pseudo(old_generator, generator)
    for epoch in range(opt.phase2_epochs):
        for (image, label), (image_, label_) in tqdm(zip(data_loader, data_loader)):
            image = image.to(opt.device)
            image_ = image_.to(opt.device)
            good_reconstruction = generator(image)
            bad_reconstruction = old_generator(image)
            pseudo_reconstruction = pseudo_generator(image, image_)
            good_score = discriminator(good_reconstruction)
            init_score = discriminator(image)
            bad_score = discriminator(bad_reconstruction)
            pseudo_score = discriminator(pseudo_reconstruction)
            loss_func = nn.BCEWithLogitsLoss()
            for param in generator.parameters():
                param.requires_grad_(False)
            for param in old_generator.parameters():
                param.requires_grad_(False)
            loss = (1 - opt.alpha) * loss_func(torch.ones_like(good_score), good_score) + \
                   opt.alpha * loss_func(torch.ones_like(init_score), init_score) +\
                   opt.beta * loss_func(torch.zeros_like(bad_score), bad_score) + \
                   (1 - opt.beta) * loss_func(torch.zeros_like(pseudo_score), pseudo_score)
            discriminator.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
        test(opt, generator, discriminator, epoch)
    print("phase 2 training finished and now saving discriminator for good reconstruction and bad one.")
    torch.save(discriminator, './models/discriminator_good_bad.pkl')
    print("discriminator for good and bad saved!!!")
    return discriminator


def train(opt):
    old_generator, generator, discriminator = train_phase_1(opt)
    # old_generator = torch.load('./models/old_generator.pkl')
    # generator = torch.load('./models/trained_generator.pkl')
    # discriminator = torch.load('./models/discriminator_real_fake.pkl')
    discriminator = train_phase_2(opt, old_generator, generator, discriminator)
    test(opt, generator, discriminator)


def test(opt, generator, discriminator, epoch=0):
    opt.drop_last = False
    opt.data_path = './data/test/'
    Data_loader = load_data(opt)
    generator = generator.to(opt.device)
    discriminator = discriminator.to(opt.device)
    labels = []
    d_results = []
    count = 0
    import random
    seed = random.randint(0, 30)
    for input, label in Data_loader:
        input = input.to(opt.device)
        g_output = generator(input)
        d_fake_output = discriminator(g_output)
        count += 1
        if count == seed:
            save_image(g_output, './imgs/phase2/{}.png'.format(epoch))
        d_results.append(d_fake_output.cpu().detach().numpy())
        labels.append(label)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    eer_threshold1 = eer_threshold1
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    print("AUC: {0}, F1_score: {1}".format(metrics.auc(fpr1, tpr1), f1_score))


if __name__ == '__main__':
    train(Opt)
