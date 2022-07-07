'''
	modified to include sha256 verification and ViGNN models
'''

# 2020.06.09-GhostNet definition for pytorch hub
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import torch
import torchvision
from ghostnet_pytorch.ghostnet import ghostnet
from vig_pytorch import pyramid_vig as _pyramid_vig
from wavemlp_pytorch.models import wavemlp as _wavemlp

dependencies = ['torch', 'torchvision']

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

	return model, transform

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

	return model, transform

def pvig_s_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_s_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_s_224_gelu()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vignn/pvig_s_82.1-03455330.pth.tar"
		cache_file_name = "pvig_s_82.1-03455330.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '03455330'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	return model, transform

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

	return model, transform

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

	return model, transform

# ===================================================================
#  wavemlp
# ===================================================================

def wavemlp_t_dw_in1k(pretrained=True, **kwargs):
	"""
	Wave MLP (wavemlp_t_dw)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _wavemlp.WaveMLP_T_dw()
	if pretrained:
		checkpoint_url = "https://github.com/huawei-noah/CV-Backbones/releases/download/wavemlp/WaveMLP_T_dw.pth.tar"
		cache_file_name = "WaveMLP_T_dw-cf09a27d.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'cf09a27d'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform
