import torch
from torch import optim
from torch.nn import init
from torch.autograd import grad
import numpy as np
import os
import time
from PIL import Image

from model import define_D , define_G
from utils import set_requires_grad




# build model

class CycleGAN():

    def __init__(self,opt):
        
        # initialize hyper-parameters
        self.cycle_weight = opt.cycle_weight
        self.idt_weight = opt.idt_weight


        self.G1 = define_G(in_dim=1,out_dim=3,conv_dim=64,norm_type="instance") #generate fake B 
        self.G2 = define_G(in_dim=3,out_dim=1,conv_dim=64,norm_type="instance") #generate fake A
        self.D1 = define_D(in_dim=1,dim=64,norm_type="instance") #judge input is B or not
        self.D2 = define_D(in_dim=3,dim=64,norm_type="instance") #judge input is A or not
        
        self.cycle_loss = torch.nn.L1Loss()
        self.idt_loss = torch.nn.L1Loss()

        self.g_optimizer = optim.RMSprop(list(self.G1.parameters())+list(self.G2.parameters()), self.lr)
        self.d_optimizer = optim.RMSprop(list(self.D1.parameters())+list(self.D2.parameters()), self.lr)
    
    def set_inputs(self,domainA,domainB):
        self.real_A = domainA
        self.real_B = domainB

    def forward(self):

        self.fake_B = self.G1(self.real_A)
        self.rec_A = self.G2(self.fake_B)
        self.fake_A = self.G2(self.real_B)
        self.rec_B = self.G1(self.fake_A)
    
    def backward_G(self):

        g1_adv_loss = -torch.mean(self.D1(self.fake_B)) 
        g2_adv_loss = -torch.mean(self.D2(self.fake_A))
        forward_cycle_loss = self.cycle_loss(self.rec_A,self.rec_A)*self.cycle_weight
        backward_cycle_loss = self.cycle_loss(self.rec_B,self.real_B)*self.cycle_weight
        g1_idt_loss = self.idt_loss(self.fake_B,self.real_A)*self.idt_weight
        g2_idt_loss = self.idt_loss(self.fake_A,self.real_B)*self.idt_weight

        self.g_loss = g1_adv_loss + g2_adv_loss + forward_cycle_loss + backward_cycle_loss + g1_idt_loss + g2_idt_loss
        self.g_loss.backward()

    def backward_D(self):

        d1_loss = -torch.mean(self.D1(self.real_B)) + torch.mean(self.D1(self.fake_B))
        d2_loss = -torch.mean(self.D2(self.real_A)) + torch.mean(self.D2(self.fake_A))

        self.d_loss = d1_loss + d2_loss
        self.d_loss.backward()

    def weight_clip(self):
        for param in self.D1.parameters():
            torch.clamp(param,-self.clip,self.clip)
        for param in self.D2.parameters():
            torch.clamp(param,-self.clip,self.clip)

    def optimize_parameters(self):
        self.forward()
        set_requires_grad([self.D1,self.D2],requires_grad=False) # uneccesery when training G
        self.g_optimizer.zero_grad()   
        self.backward_G()
        self.g_optimizer.step()
        set_requires_grad([self.D1,self.D2],True)
        self.d_optimizer.zero_grad()
        self.backward_D()
        self.d_optimizer.step()
        self.weight_clip()
