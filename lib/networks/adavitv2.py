import math
import time
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_3tuple
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_, lecun_normal_
from timm.models.helpers import named_apply
from timm.models.vision_transformer import Block as oldBlock

from .gumbel_tricks import gumbel_soft_topk

import pdb

from .adavit import init_weights_vit_timm, init_weights_vit_jax, init_weights_vit_moco, \
    get_init_weights_vit, MlpPredictor, AttPredictor, Attention, LayerScale, \
    Block, MaskCompletion, ConcatUnshuffle, CompletionNet, FusionNet, FusionNetV2

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

class VisionTransformerTokenPruningV2(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, 
            predictor=MlpPredictor, tp_loc=None, tp_ratio=None, tp_tau=1., distill=False, as_encoder=True, args=None):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.args = args

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.as_encoder = as_encoder
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.policy_list = nn.ParameterList([torch.tensor(range(num_patches + self.num_tokens)) for _ in range(len(tp_loc))], 
                                                 requires_grad=False)
        # self.policy.requires_grad = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        # score predictor for token prediction
        self.tp_loc = tp_loc
        self.tp_ratio = tp_ratio
        self.tp_tau = tp_tau
        self.distill = distill
        self.score_predictor_list = nn.ModuleList([
            predictor(embed_dim) for _ in range(len(tp_loc))
        ])
        self.mid_norm_list = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(len(tp_loc))
        ]) if as_encoder else None

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def get_num_layers(self):
        return len(self.blocks)

    def prune_token(self, x, score_predictor, global_kept_index=None, time_meters=None):
        B, Lx, C = x.shape
        L = Lx - 1
        pred_score = score_predictor(x).reshape(B, L) # [B, L]. The [CLS] should be removed
        keep_k = int(L * (1 - self.tp_ratio))
        if self.training:
            raise NotImplementedError("training phase for v2diff pruning is not yet implemented")
            # # compute current policy
            # cur_policy = gumbel_soft_topk(pred_score, keep_k, tau=self.tp_tau, hard=True) # cur_policy of size [B, L]
            # cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype), 
            #                         cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
            # # pruning tokens of x according to cur_policy
            # kept_x = x[cur_policy.bool()].reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, C]
            # pruned_x = x[~cur_policy.bool()].reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
            # # update the global policy
            # keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
            # keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
            # prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
            # # For backpropagating gradients
            # diff_keep_policy = keep_policy[keep_policy.bool()].reshape(B, keep_k + 1, 1)
            # diff_prune_policy = prune_policy[prune_policy.bool()].reshape(B, L - keep_k, 1)
            # kept_x = kept_x * diff_keep_policy
            # pruned_x = pruned_x * diff_prune_policy
        else:
            # current keep token indices
            s_time = time.perf_counter()

            sort_index = pred_score.sort(descending=True)[1]
            keep_noncls_idx = sort_index[:, :keep_k] + 1
            # keep_noncls_idx = pred_score.topk(keep_k, dim=-1)[1] + 1 # [B, keep_k]
            kept_index = torch.cat([torch.zeros(B, 1, dtype=keep_noncls_idx.dtype, device=keep_noncls_idx.device), 
                                  keep_noncls_idx], 
                                  dim=1)
            # pruned_index = (-pred_score).topk(L - keep_k, dim=-1)[1] + 1
            pruned_index = sort_index[:, keep_k:] + 1

            # keep_noncls_idx = 

            # pdb.set_trace()
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2topk'].append(time.perf_counter() - s_time)
            # build current policy
            # cur_policy = torch.zeros_like(pred_score).scatter_(-1, keep_noncls_idx, 1.0) # cur_policy of size [B, L]
            # s_time = time.perf_counter()
            # cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype), 
            #                         cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
            # if time_meters is not None:
            #     time_meters[f'{L}_cat'].append(time.perf_counter() - s_time)
            # pruning tokens of x according to cur_policy
            # s_time = time.perf_counter()
            # kept_index = (cur_policy.bool()).nonzero(as_tuple=False)[:, 1].view(B, keep_k + 1)
            # pruned_index = ~(cur_policy.bool()).nonzero(as_tuple=False)[:, 1].view(B, L - keep_k)
            # if time_meters is not None:
            #     time_meters[f'{L}_2getidx'].append(time.perf_counter() - s_time)
            s_time = time.perf_counter()
            kept_x = torch.gather(x, dim=1, index=kept_index[:, :, None].expand(-1, -1, C))
            pruned_x = torch.gather(x, dim=1, index=pruned_index[:, :, None].expand(-1, -1, C))
            # pdb.set_trace()
            # kept_x = x[cur_policy.bool()].reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, 1]
            # pruned_x = x[~cur_policy.bool()].reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2gather_x'].append(time.perf_counter() - s_time)
            # update the global policy
            s_time = time.perf_counter()
            if global_kept_index is None:
                global_kept_index, global_pruned_index = kept_index, pruned_index
            else:
                global_kept_index, global_pruned_index = torch.gather(global_kept_index, dim=1, index=kept_index), \
                    torch.gather(global_kept_index, dim=1, index=pruned_index)
            # keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
            # keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
            # prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2gather_index'].append(time.perf_counter() - s_time)
            # pdb.set_trace()
        return kept_x, pruned_x, global_kept_index, global_pruned_index

    def forward_features(self, x, time_meters=None):
        args = self.args
        s_time = time.perf_counter()
        x = self.patch_embed(x)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['patchembed'].append(time.perf_counter() - s_time)
        B, L, _ = x.shape
        
        s_time = time.perf_counter()
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['cls+pe'].append(time.perf_counter() - s_time)

        # policy = torch.ones(B, L + 1, dtype=x.dtype, device=x.device) # [B, L + 1]
        s_time = time.perf_counter()
        # policy = self.policy[None, :].expand(B, -1)
        policy = None
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['init_policy'].append(time.perf_counter() - s_time)
        policy_list = []
        token_list = []
        for i, blk in enumerate(self.blocks):
            s_time = time.perf_counter()
            x = blk(x)
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'block{i+1}'].append(time.perf_counter() - s_time)
            if i in self.tp_loc:
                s_time = time.perf_counter()
                j = self.tp_loc.index(i)
                score_predictor = self.score_predictor_list[j] if not args.share_pred else self.score_predictor_list[0]
                x, pruned_x, policy, prune_policy = self.prune_token(x, score_predictor, global_kept_index=policy, time_meters=None)
                token_list.append(self.mid_norm_list[j](pruned_x))
                policy_list.append(prune_policy.detach())
                if time_meters is not None:
                    torch.cuda.synchronize()
                    time_meters[f'{j}_tp'].append(time.perf_counter() - s_time)
        token_list.append(self.norm(x))
        policy_list.append(policy.detach())
        
        return token_list, policy_list

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, time_meters=None):
        token_list, policy_list = self.forward_features(x, time_meters=time_meters)
        if self.as_encoder:
            return token_list, policy_list
        else:
            x = self.forward_head(token_list[-1])
            return x

if __name__ == "__main__":
    model = VisionTransformerTokenPruningV2(num_classes=0, global_pool='', tp_loc=[2, 5, 8], tp_ratio=0.25)
    input_tensor = torch.zeros(2, 3, 224, 224)
    # pdb.set_trace()
    output_tensor, policy_list = model(input_tensor)
    pdb.set_trace()