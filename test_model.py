import torch

from basicsr.models.archs.fftformer_arch import fftformer

checkpoint = torch.load('./pretrain_model/net_g_GoPro_HIDE.pth')

net = fftformer()

net.load_state_dict(checkpoint,state_dict=False)

torch.save(net.state_dict(),'./pretrain_model/fftformer_GoPro_HIDE.pth')