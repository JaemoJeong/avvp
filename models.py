import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
    def __call__(self, labels, vision_transformed, audio_transformed, video_id, similarity_type='audio', vision_mode='video', start_time=0, end_time=10, audio_transformed_full=None):
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

            # Global Audio Embeddings (if full 10s audio provided)
            if audio_transformed_full is not None:
                languagebind_inputs_global = {'language': languagebind_inputs['language']}
                languagebind_inputs_global['audio'] = {"pixel_values": audio_transformed_full}
                with torch.no_grad():
                    def compute_global_audio():
                        return self.model(languagebind_inputs_global)['audio']
                    global_audio_feat = self.cache.get_or_compute(video_id, 'global_audio', compute_global_audio, start_time, end_time).to(self.device)
                similarities['global_audio'] = norm_similarities(global_audio_feat @ text_emb.T)
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
    def __init__(self, device, cache_enabled=True, use_tci=False, use_cross_modal=False):
        self.device = device
        self.use_cross_modal = use_cross_modal
        # self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False).to(self.device)
        self.clap.load_ckpt()
        self.name = "ClipClap"
        self.use_tci = use_tci
        self.cache = EmbeddingCache(self.name, enabled=cache_enabled)

        # Temporal Context Injection — full self-attention (Identity projections)
        if self.use_tci:
            from temporal_context import TemporalContextInjector

            self.vision_tci = TemporalContextInjector(d=1024).to(device)   # ViT-L/14 pre-proj dim
            self.audio_tci = TemporalContextInjector(d=768).to(device)     # CLAP HTSAT dim
            print("[TCI] Temporal Context Injectors initialized (Identity, full self-attention)")

    def _encode_image_pre_proj(self, images):
        """CLIP ViT forward → pre-projection features (e.g. 1024 for ViT-L/14)"""
        visual = self.clip_model.visual
        x = visual.conv1(images.type(self.clip_model.dtype))
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
        x = visual.ln_post(x[:, 0, :])
        return x  # (B, 1024)

    def _encode_audio_pre_proj(self, audio_data):
        """CLAP HTSAT forward → pre-projection features (e.g. 768)"""
        from laion_clap.training.data import get_audio_features

        audio_input = []
        for i in range(audio_data.shape[0]):
            waveform = audio_data[i]
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float()
            temp_dict = get_audio_features(
                {}, waveform, 480000,
                data_truncating='rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.clap.model_cfg['audio_cfg'],
                require_grad=False
            )
            audio_input.append(temp_dict)

        input_dict = {}
        for k in audio_input[0].keys():
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_input], dim=0).to(self.device)

        audio_embeds = self.clap.model.encode_audio(input_dict, device=self.device)["embedding"]
        return audio_embeds  # (B, 768)

    def __call__(self, labels, vision_transformed=None, audio_transformed=None, video_id=None, similarity_type='audio', vision_mode='video', start_time=0, end_time=10, audio_transformed_full=None):
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

                if self.use_tci:
                    audio_pre = self._encode_audio_pre_proj(audio_transformed)        # (T, 768)
                    enriched = self.audio_tci(audio_pre)                              # (T, 768) — already f + g*V
                    enriched_proj = self.clap.model.audio_projection(enriched)        # (T, 512)
                    audio_feat = enriched_proj
                
                # ALWAYS L2 normalize audio_feat
                audio_feat = F.normalize(audio_feat, dim=-1)
                
                clap_text_feat = self.clap.get_text_embedding(clap_text_labels, use_tensor=True).to(self.device)
                clap_text_feat = F.normalize(clap_text_feat, dim=-1)              # L2 normalize text features
                
                raw_audio_logits = audio_feat @ clap_text_feat.T              # (T, K) — raw cosine similarity
                
                # Store the true raw logits for export before any guidance
                similarities['audio_raw'] = raw_audio_logits.clone()
                print(f"[RAW] audio logits: min={raw_audio_logits.min():.3f}, max={raw_audio_logits.max():.3f}, mean={raw_audio_logits.mean():.3f}")

                if audio_transformed_full is not None:
                    def compute_global_audio():
                        return self.clap.get_audio_embedding_from_data(x=audio_transformed_full, use_tensor=True)
                    global_audio_feat = self.cache.get_or_compute(video_id, 'global_audio', compute_global_audio, start_time, end_time).to(self.device)
                    global_audio_feat = F.normalize(global_audio_feat, dim=-1) # L2 normalize global audio
                    raw_global_audio_logits = global_audio_feat @ clap_text_feat.T
                    similarities['global_audio'] = norm_similarities(raw_global_audio_logits)
            # Vision
            if ('image' in similarity_type or 'video' in similarity_type) and vision_transformed is not None:
                def compute_vision():
                    return self.clip_model.encode_image(vision_transformed)
                vision_feat = self.cache.get_or_compute(video_id, vision_mode, compute_vision, start_time, end_time).to(self.device)

                if self.use_tci:
                    vision_pre = self._encode_image_pre_proj(vision_transformed)      # (T, 1024)
                    enriched = self.vision_tci(vision_pre)                             # (T, 1024) — f + g*V
                    vision_feat = enriched @ self.clip_model.visual.proj              # (T, 768)
                
                # ALWAYS L2 normalize vision_feat to bounded cosine similarity
                vision_feat = F.normalize(vision_feat, dim=-1)
                
                clip_text_feat = self.clip_model.encode_text(clip_text_labels)
                clip_text_feat = F.normalize(clip_text_feat, dim=-1)              # L2 normalize text features
                vision_feat = vision_feat.to(clip_text_feat.dtype)
                
                raw_vision_logits = vision_feat @ clip_text_feat.T            # (T, K) — raw cosine similarity
                
                # Store the true raw logits for export before any guidance
                similarities['image_raw'] = raw_vision_logits.clone()
                print(f"[RAW] vision logits: min={raw_vision_logits.min():.3f}, max={raw_vision_logits.max():.3f}, mean={raw_vision_logits.mean():.3f}")
                
                if vision_mode == 'image':
                    similarities['image_features'] = vision_feat

            # ── Audio-Visual Cross-Modal Guidance on RAW logits ──
            has_audio = 'raw_audio_logits' in dir()
            has_vision = 'raw_vision_logits' in dir()
            
            if has_audio and has_vision and self.use_cross_modal:
                # Video-level prior from RAW logits
                P_audio_global = F.softmax(raw_audio_logits.float().mean(dim=0), dim=-1)    # (K,)
                P_visual_global = F.softmax(raw_vision_logits.float().mean(dim=0), dim=-1)  # (K,)
                
                # Cross-modal guidance on raw logits
                raw_vision_logits = raw_vision_logits + P_audio_global.unsqueeze(0)
                raw_audio_logits = raw_audio_logits + P_visual_global.unsqueeze(0)
                
                print(f"[AV-Guide] video-level prior: P_audio top={P_audio_global.max():.3f}, P_visual top={P_visual_global.max():.3f}")
            
            # ── Final norm_similarities (only once, after all modifications) ──
            if has_audio:
                similarities['audio'] = norm_similarities(raw_audio_logits)
            if has_vision:
                similarities[vision_mode] = norm_similarities(raw_vision_logits)
                if vision_mode == 'video':
                    similarities['video'] = similarities['video'].mean(dim=0, keepdim=True)

        return similarities
