from network.pspnet_origion import PSPNet
from network.deeplab import Deeplab
from network.refinenet import RefineNet


from network.relighting_origion import LightNet, L_TV, L_exp_z, SSIM
from network.discriminator_SE import FCDiscriminator
from network.loss import StaticLoss, SCILoss
from network.relighting_SCI import EnhanceNetwork2, EnhanceNetwork
from network.relighting_SCIc_net import Network

# from network.SCImodel import Network, Finetunemodel, EnhanceNetwork
# from network.zeroDCE import enhance_net_nopool
# from network.pspnet_aspp import PSPNetASPP
# from network.pspnet_inception import PSPNetInception
# from network.unet import UNet