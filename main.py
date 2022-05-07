import argparse
import os
import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/prediction/')
from natsort import natsorted
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from lib import Pre_dataset
from  network import  Seq2seqGRU
from config import ParseGRU
from torch.utils.data import DataLoader

parse  = ParseGRU()
opt    = parse.args

autoencoder = Seq2seqGRU(opt)
autoencoder.train()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=opt.learning_rate,
                             weight_decay=1e-5)
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(1),
    transforms.Resize((opt.image_size, opt.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# has all image shape,[data,label]
datat_ = Pre_dataset(opt, opt.dataset, extension=opt.img_extension, transforms=transform)  # b,label?
train_loader = DataLoader(datat_, batch_size=opt.batch_size, shuffle=True)  # if shuffle

# has all image shape,[data,label]
datatest_ = Pre_dataset(opt, opt.testset, extension=opt.img_extension, transforms=transform)  # b,label?,T
test_loader = DataLoader(datatest_, batch_size=1, shuffle=False)  # if shu

if opt.cuda:
    autoencoder.cuda()

losses = np.zeros(opt.num_epochs)

for itr in range(opt.num_epochs):
    autoencoder.train()
    for data, ydata in train_loader:

        if data.size(0) != opt.batch_size:
            break

        x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size, opt.image_size)
        y = ydata.reshape(-1, opt.n_channels, opt.image_size, opt.image_size)

        if opt.cuda:
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x)

        yhat = autoencoder(x)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = mse_loss(yhat, y)
        losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss: {:.4f}'.format(
        itr + 1,
        opt.num_epochs,
        loss))


    if itr % opt.check_point == 0:
        autoencoder.eval()
        for data, ydata in test_loader:

            if data.size(0) != opt.batch_size:
                break

            x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size, opt.image_size)
            y = ydata.reshape(-1, opt.n_channels, opt.image_size, opt.image_size)

            if opt.cuda:
                x = Variable(x).cuda()
                y = Variable(y).cuda()
            else:
                x = Variable(x)

            yhat = autoencoder(x)

        tests = y[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)
        recon = yhat[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size, opt.image_size)

        for i in range(opt.n_test):
            # if itr == 0:
            save_image((tests[i] / 2 + 0.5),
                       os.path.join(opt.log_folder + '/generated_videos', "real_itr{}_no{}.png".format(itr, i)))
            save_image((recon[i] / 2 + 0.5),
                       os.path.join(opt.log_folder + '/generated_videos', "recon_itr{}_no{}.png".format(itr, i)))
            # torch.save(autoencoder.state_dict(), os.path.join('./weights', 'G_itr{:04d}.pth'.format(itr+1)))

