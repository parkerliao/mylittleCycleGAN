import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import time
from qqdm.notebook import qqdm
from PIL import Image

from cycle_gan import CycleGAN
from dataset import CycleGANDataSet
from utils import merge_images


def train(config):

    # define transform

    transform = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.RandomCrop((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.toTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])

    # load dataset

    dataset = CycleGANDataSet(config.data_path1, config.data_path2, transform)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True)

    model = CycleGAN(config)

    for epoch in config.epochs:

        progress_bar = qqdm(dataloader)
        steps = 0
        for i, data in enumerate(progress_bar, 0):

            if torch.cuda().is_available():
                img_A = data[0]
                img_B = data[1]
                img_A = img_A.cuda()
                img_B = img_B.cuda()

            model.set_inputs(img_A, img_B)
            model.optimize_parameters()

            progress_bar.set_infos({
                'loss_D': round(model.d_loss.item(), 4),
                'loss_G': round(model.g_loss.item(), 4),
                'Epoch': epoch+1,
            })

        merged = merge_images(model.real_A, model.fake_B)
        merged = Image.fromarray(merged)
        merged.save(config.log_path+'AtoB-'+str(epoch+1)+'.png')

        merged = merge_images(model.real_B, model.fake_A)
        merged = Image.fromarray(merged)
        merged.save(config.log_path+'BtoA-'+str(epoch+1)+'.png')

        if (epoch+1) % 10 == 0:

            torch.save(model.G1.state_dict(), config.model_path +
                       'G1-'+str(epoch+1)+'.pkl')
            torch.save(model.G2.state_dict(), config.model_path +
                       'G2-'+str(epoch+1)+'.pkl')
            torch.save(model.D1.state_dict(), config.model_path +
                       'D1-'+str(epoch+1)+'.pkl')
            torch.save(model.D2.state_dict(), config.model_path +
                       'D2-'+str(epoch+1)+'.pkl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # path

    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--data_path1', type=str)
    parser.add_argument('--data_path2', type=str)

    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--cycle_weight', type=float, default=10)
    parser.add_argument('--ldt_weight', type=float, default=0.5)

    config = parser.parse_args()
