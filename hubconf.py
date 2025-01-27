# 2022.09.16-GhostNet & SNN-MLP definition for pytorch hub
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import torch
import torchvision
from ghostnet_pytorch.ghostnet import ghostnet
from vig_pytorch import pyramid_vig as _pyramid_vig
from vig_pytorch import vig as _vig
from wavemlp_pytorch.models import wavemlp as _wavemlp
from pdb import set_trace

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

def vig_b_224_gelu(pretrained=True, **kwargs):
    """
	Vision GNN (vig_b_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
    model = _vig.vig_b_224_gelu(**kwargs)
    if pretrained:
        checkpoint_url = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/vig/vig_b_82.6.pth'
        cache_file_name = "vig_b_82.6-40b0685d.pth"
        state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
        )
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
        model.hashid = '40b0685d'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
        
    transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model

def vig_s_224_gelu(pretrained=True, **kwargs):
    """
	Vision GNN (vig_s_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
    model = _vig.vig_s_224_gelu(**kwargs)
    if pretrained:
        checkpoint_url = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/vig/vig_s_80.6.pth'
        cache_file_name = "vig_s_80.6-081bb44c.pth"
        state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
        model.hashid = '081bb44c'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
        
    transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model

def vig_ti_224_gelu(pretrained=True, **kwargs):
    """
	Vision GNN (vig_b_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
    model = _vig.vig_ti_224_gelu(**kwargs)
    if pretrained:
        checkpoint_url = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/vig/vig_ti_74.5.pth'
        cache_file_name = "vig_ti_74.5-61872146.pth"
        state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
        model.hashid = '61872146'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
        
    transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model
    
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
		msg = model.load_state_dict(state_dict, strict=True)
		print(msg)
		model.hashid = '06c49bda'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	return model

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
		msg = model.load_state_dict(state_dict, strict=True)
		print(msg)
		model.hashid = '03455330'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
	transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
		msg = model.load_state_dict(state_dict, strict=True)
		print(msg)
		model.hashid = '5a5ce0c0'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
	transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model

def pvig_b_224_gelu_in1k(pretrained=True, **kwargs):
	"""
	Vision GNN (pvig_b_224_gelu)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _pyramid_vig.pvig_b_224_gelu()
	if pretrained:
		checkpoint_url = "https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vignn/pvig_b_83.66-aafa414a.pth.tar"
		cache_file_name = "pvig_b_83.66-aafa414a.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		msg = model.load_state_dict(state_dict, strict=True)
		print(msg)
		model.hashid = 'aafa414a'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
	transform = _transform(resize=int(224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
	return model

# ===================================================================
#  others...
# ===================================================================

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
    from snnmlp_pytorch.models.snn_mlp import SNNMLP as _SNNMLP
	model = _SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], drop_path_rate=0.2)
	if pretrained:
		state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_t, progress=True)
		model.load_state_dict(state_dict)
	return model

def snnmlp_s(pretrained=False, **kwargs):
	""" # This docstring shows up in hub.help()
    SNN-MLP small model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from snnmlp_pytorch.models.snn_mlp import SNNMLP as _SNNMLP
	model = _SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 18, 2], drop_path_rate=0.3)
	if pretrained:
		state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_s, progress=True)
		model.load_state_dict(state_dict)
	return model

def snnmlp_b(pretrained=False, **kwargs):
	""" # This docstring shows up in hub.help()
    SNN-MLP base model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from snnmlp_pytorch.models.snn_mlp import SNNMLP as _SNNMLP
	model = _SNNMLP(num_classes=1000, embed_dim=128, depths=[2, 2, 18, 2], drop_path_rate=0.5)
	if pretrained:
		state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_b, progress=True)
		model.load_state_dict(state_dict)
	return model

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
		state_dict = {k:v for k,v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params") }
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'cf09a27d'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def wavemlp_t_in1k(pretrained=True, **kwargs):
	"""
	Wave MLP (wavemlp_t)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _wavemlp.WaveMLP_T()
	if pretrained:
		checkpoint_url = "https://github.com/huawei-noah/CV-Backbones/releases/download/wavemlp/WaveMLP_T.pth.tar"
		cache_file_name = "WaveMLP_T-6b42f045.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k:v for k,v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params") }
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '6b42f045'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def wavemlp_s_in1k(pretrained=True, **kwargs):
	"""
	Wave MLP (wavemlp_s)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _wavemlp.WaveMLP_S()
	if pretrained:
		checkpoint_url = "https://github.com/huawei-noah/CV-Backbones/releases/download/wavemlp/WaveMLP_S.pth.tar"
		cache_file_name = "WaveMLP_S-1a1f39fc.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k:v for k,v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params") }
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '1a1f39fc'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def wavemlp_m_in1k(pretrained=True, **kwargs):
	"""
	Wave MLP (wavemlp_m)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = _wavemlp.WaveMLP_M()
	if pretrained:
		checkpoint_url = "https://github.com/huawei-noah/CV-Backbones/releases/download/wavemlp/WaveMLP_M.pth.tar"
		cache_file_name = "WaveMLP_M-39dd2019.pth.tar"
		state_dict = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k:v for k,v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params") }
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '39dd2019'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=int(224/.9), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

