import torch
import torch.nn as nn
from languagebind.languagebindmodel.languagebind import LanguageBind, LanguageBindImageTokenizer
import clip
import laion_clap
from embedding_cache import EmbeddingCache

def pad_similarities(vision_text_similarity, audio_text_similarity, device):
    if vision_text_similarity.shape[0] < audio_text_similarity.shape[0]:
        pad = torch.zeros_like(vision_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat_interleave(audio_text_similarity.shape[0] - vision_text_similarity.shape[0])
        vision_text_similarity = torch.cat((vision_text_similarity, pad), dim=0)
    elif vision_text_similarity.shape[0] > audio_text_similarity.shape[0]:
        pad = torch.zeros_like(audio_text_similarity[0].unsqueeze(0)).to(device)
        pad = pad.repeat(vision_text_similarity.shape[0] - audio_text_similarity.shape[0], 1)
        audio_text_similarity = torch.cat((audio_text_similarity, pad), dim=0)
    
    return vision_text_similarity, audio_text_similarity

def norm_similarities(similarities):
    similarities = (similarities - torch.mean(similarities, dim=-1, keepdim=True)) / torch.std(similarities, dim=-1, keepdim=True)
    similarities = torch.sigmoid(similarities)
    return similarities

class LanguageBind_model:
    def __init__(self, device, cache_enabled=True):
        clip_type = {
            'video': 'LanguageBind_Video_FT', 
            'audio': 'LanguageBind_Audio_FT',
            'image': 'LanguageBind_Image',
        }
        self.device = device
        self.model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.model = self.model.to(self.device)
        self.model.eval()
        pretrained_ckpt = f'lb203/LanguageBind_Image'
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
        self.name = 'LanguageBind'
        self.cache = EmbeddingCache(self.name, enabled=cache_enabled)
        
    def __call__(self, labels, vision_transformed, audio_transformed, video_id, similarity_type='audio', vision_mode='video', start_time=0, end_time=10):
        similarities = {}
        
        # Text Embeddings
        preprocessed_labels = [f"A {label.replace('_', ' ').lower()}" for label in labels]
        languagebind_inputs = {'language': self.tokenizer(preprocessed_labels, max_length=77, padding='max_length',
                                                          truncation=True, return_tensors='pt').to(self.device)}
        with torch.no_grad():
            text_emb = self.model(languagebind_inputs)['language']

        # Audio Embeddings
        if ('audio' in similarity_type) and audio_transformed is not None:
            languagebind_inputs['audio'] = {"pixel_values": audio_transformed}
            with torch.no_grad():
                # For per-second audio (10 chunks), we might want to cache individually or as a batch.
                # get_or_compute expects start/end. 
                # If audio_transformed has batch > 1 (e.g. 10), it means we are processing the whole video in 1s chunks.
                # In that case, we can loop or assume the user wants the full tensor cached (which might be large/complicated with current simple cache).
                # However, original code used get_or_put_embedding with 'prefix' for batch > 1.
                # Here we will simplify: if batch > 1, we assume it's 0-10s.
                
                def compute_audio():
                    return self.model(languagebind_inputs)['audio']

                # Use a specific logic: if batch > 1, cache as 'batch' or similar?
                # Original code: if batch > 1, prefix='parallel', start=start_time, end=end_time.
                # We will just pass start_time, end_time to cache.
                audio_feat = self.cache.get_or_compute(video_id, 'audio', compute_audio, start_time, end_time).to(self.device)
            
            similarities['audio'] = norm_similarities(audio_feat @ text_emb.T)

        # Vision Embeddings
        if ('image' in similarity_type or 'video' in similarity_type) and vision_transformed is not None:
            # Determine processing key based on request or default
            sim_key = vision_mode if vision_mode else ('image' if 'image' in similarity_type else 'video')
            
            languagebind_inputs[sim_key] = {"pixel_values": vision_transformed}
            
            with torch.no_grad():
                def compute_vision():
                    return self.model(languagebind_inputs)[sim_key]
                
                # Careful with caching key logic.
                vision_feat = self.cache.get_or_compute(video_id, sim_key, compute_vision, start_time, end_time).to(self.device)

            similarities[sim_key] = norm_similarities(vision_feat @ text_emb.T)
            
            if sim_key == 'image':
                similarities['image_features'] = vision_feat

        return similarities


class CLIP_CLAP_model:
    def __init__(self, device, cache_enabled=True):
        self.device = device
        # self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False).to(self.device)
        self.clap.load_ckpt()
        self.name = "ClipClap"
        self.cache = EmbeddingCache(self.name, enabled=cache_enabled)

    def __call__(self, labels, vision_transformed=None, audio_transformed=None, video_id=None, similarity_type='audio', vision_mode='video', start_time=0, end_time=10):
        similarities = {}
        
        # Text Embeddings
        clap_text_labels = [f"This is a sound of {label}" for label in labels]
        clip_text_labels = clip.tokenize([f"A {label}" for label in labels]).to(self.device)

        with torch.no_grad():
            # Audio
            if ('audio' in similarity_type) and audio_transformed is not None:
                def compute_audio():
                    return self.clap.get_audio_embedding_from_data(x=audio_transformed, use_tensor=True)
                
                audio_feat = self.cache.get_or_compute(video_id, 'audio', compute_audio, start_time, end_time).to(self.device)
                
                clap_text_feat = self.clap.get_text_embedding(clap_text_labels, use_tensor=True).to(self.device)
                similarities['audio'] = norm_similarities(audio_feat @ clap_text_feat.T)

            # Vision
            if ('image' in similarity_type or 'video' in similarity_type) and vision_transformed is not None:
                def compute_vision():
                    return self.clip_model.encode_image(vision_transformed)
                
                vision_feat = self.cache.get_or_compute(video_id, vision_mode, compute_vision, start_time, end_time).to(self.device)
                
                clip_text_feat = self.clip_model.encode_text(clip_text_labels)
                vision_feat = vision_feat.to(clip_text_feat.dtype) # match dtype
                
                similarities[vision_mode] = norm_similarities(vision_feat @ clip_text_feat.T)
                
                if vision_mode == 'image':
                    similarities['image_features'] = vision_feat
                if vision_mode == 'video':
                    similarities['video'] = similarities['video'].mean(dim=0, keepdim=True)

        return similarities
