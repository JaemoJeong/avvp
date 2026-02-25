"""Verify _encode_image_pre_proj and _encode_audio_pre_proj are exactly pre-projection"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '/home/jaemo/AVVP')
import torch
import torch.nn.functional as F
import clip
import laion_clap

device = 'cuda:1'

# ======== CLIP Vision ========
print("=" * 50)
print("CLIP ViT-L/14 검증")
print("=" * 50)

model, _ = clip.load('ViT-L/14', device=device)
img = torch.randn(2, 3, 224, 224).to(device)

# 기존 encode_image
feat_standard = model.encode_image(img)

# pre_proj → 수동 projection
visual = model.visual
x = visual.conv1(img.type(model.dtype))
x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
x = torch.cat([
    visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
    x
], dim=1)
x = x + visual.positional_embedding.to(x.dtype)
x = visual.ln_pre(x)
x = x.permute(1, 0, 2)
x = visual.transformer(x)
x = x.permute(1, 0, 2)
pre_proj = visual.ln_post(x[:, 0, :])

feat_manual = pre_proj @ model.visual.proj

diff = (feat_standard - feat_manual).abs().max().item()
print(f"encode_image:    {feat_standard.shape}")
print(f"pre_proj:        {pre_proj.shape}")
print(f"pre_proj @ proj: {feat_manual.shape}")
print(f"max diff:        {diff:.2e}")
print(f"일치: {torch.allclose(feat_standard, feat_manual, atol=1e-5)}")

# ======== CLAP Audio ========
print()
print("=" * 50)
print("CLAP HTSAT 검증")
print("=" * 50)

clap = laion_clap.CLAP_Module(enable_fusion=False).to(device)
clap.load_ckpt()

# 랜덤 오디오
audio = torch.randn(2, 480000).to(device)

# 기존 get_audio_embedding_from_data
feat_standard = clap.get_audio_embedding_from_data(x=audio, use_tensor=True)

# pre_proj → 수동 projection
from laion_clap.training.data import get_audio_features
audio_input = []
for i in range(audio.shape[0]):
    temp_dict = get_audio_features(
        {}, audio[i].cpu(), 480000,
        data_truncating='rand_trunc',
        data_filling='repeatpad',
        audio_cfg=clap.model_cfg['audio_cfg'],
        require_grad=False
    )
    audio_input.append(temp_dict)

input_dict = {}
for k in audio_input[0].keys():
    input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_input], dim=0).to(device)

pre_proj = clap.model.encode_audio(input_dict, device=device)["embedding"]
feat_manual = clap.model.audio_projection(pre_proj)
feat_manual = F.normalize(feat_manual, dim=-1)

diff = (feat_standard - feat_manual).abs().max().item()
print(f"get_audio_embedding: {feat_standard.shape}")
print(f"pre_proj:            {pre_proj.shape}")
print(f"proj + norm:         {feat_manual.shape}")
print(f"max diff:            {diff:.2e}")
print(f"일치: {torch.allclose(feat_standard, feat_manual, atol=1e-5)}")
