# 2022.09.16-GhostNet & SNN-MLP definition for pytorch hub
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import torch
import torchvision
from ghostnet_pytorch.ghostnet import ghostnet
from snnmlp_pytorch.models.snn_mlp import SNNMLP
from vig_pytorch import pyramid_vig as _pyramid_vig

dependencies = ['torch', 'torchvision']

state_dict_url = 'https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth'
state_dict_url_snnmlp_t = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_tiny_81.88.pt'
state_dict_url_snnmlp_s = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_small_83.30.pt'
state_dict_url_snnmlp_b = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_base_83.59.pt'

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

# ===================================================================
#  Ghostnet
# ===================================================================

def ghostnet_1x(pretrained=True, **kwargs):
	"""
	GhostNet 1.0x model
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
	if pretrained:
		checkpoint_url = "https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth"
		cache_file_name = "ghostnet_1x_state_dict_73.98-965143b0.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '965143b0'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model

# ===================================================================
#  Vision Graph Neural Network
# ===================================================================

def pvig_ti_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_ti_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_ti_224_gelu()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vignn/pvig_ti_78.5-06c49bda.pth.tar"
		cache_file_name = "pvig_ti_78.5-06c49bda.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '06c49bda'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	return model

def pvig_s_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_s_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_s_224_gelu()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vignn/ptnt_s_82.0-1752a427.pth.tar"
		cache_file_name = "ptnt_s_82.0-1752a427.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '1752a427'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	return model

def pvig_m_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_m_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_m_224_gelu()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vignn/pvig_m_83.1-5a5ce0c0.pth.tar"
		cache_file_name = "pvig_m_83.1-5a5ce0c0.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '5a5ce0c0'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	return model

def pvig_b_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_b_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_b_224_gelu()
	if pretrained:
		# trouble downloading weights from baidu, update when sucessfully download weights
		pass
		
	transform = _transform(resize=int(224/.95), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	return model

def ghostnet_1x(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    GhostNet 1.0x model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_t(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP tiny model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], drop_path_rate=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_t, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_s(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP small model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 18, 2], drop_path_rate=0.3)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_s, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_b(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP base model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=128, depths=[2, 2, 18, 2], drop_path_rate=0.5)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_b, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

