# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vit import PatchEmbed, Block, DecoderBlock
from einops import rearrange
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.ar_enc2dec = nn.Linear(4 * embed_dim, decoder_depth * decoder_embed_dim)
        self.kd_enc2dec = nn.Linear(4 * embed_dim, decoder_depth * decoder_embed_dim)
        self.ar_enc_norm = nn.ModuleList([nn.LayerNorm(embed_dim)]*4)
        self.kd_enc_norm = nn.ModuleList([nn.LayerNorm(embed_dim)]*4)
        self.decoder_embed_dim=decoder_embed_dim
        self.decoder_depth = decoder_depth
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.register_buffer("mask", self.mask_generate(4 - 1, 49))
        self.kd_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.ar_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                         norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.ar_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.kd_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.ar_norm = norm_layer(decoder_embed_dim)
        self.ar_pred = nn.Linear(decoder_embed_dim, 512, bias=True) # decoder to patch
        self.kd_norm = norm_layer(decoder_embed_dim)
        self.kd_pred = nn.Linear(decoder_embed_dim, 512, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        trunc_normal_(self.ar_token, std=.02)
        trunc_normal_(self.kd_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h, w, p**2 * 3))
        return x

    def mask_generate(self, segment, tokens_per_segment):
        mask = torch.tril(torch.ones((segment, segment), dtype=torch.float))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=1)
        return mask

    def raster(self, x, pos_embedding, decoder_pos_embed, label):
        # x   B h w C
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        label = rearrange(label, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        pos_embedding= rearrange(pos_embedding, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        decoder_pos_embed = rearrange(decoder_pos_embed, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        bz, num_seg, seg_size, c = x.shape
        raster_matrix = torch.stack([torch.randperm(num_seg) for _ in range(bz)], dim=0)
        raster_matrix = raster_matrix[:,:,None,None].repeat(1,1,seg_size, c).to(x.device)
        x_shuffled = torch.gather(x, dim=1, index=raster_matrix)
        label_shuffled = torch.gather(label, dim=1, index=raster_matrix[:,:,:,0:1].repeat(1,1,1,label.shape[-1]))
        pos_embedding = torch.gather(pos_embedding, dim=1, index=raster_matrix)
        decoder_pos_embed = torch.gather(decoder_pos_embed, dim=1, index=raster_matrix[:,:,:,0:1].repeat(1,1,1,decoder_pos_embed.shape[-1]))
        return x_shuffled[:, :-1], pos_embedding, decoder_pos_embed, label_shuffled
        # x_unshuffled = torch.zeros_like(x_shuffled)
        # indices = idx.unsqueeze(dim=-1).expand(-1, -1, -1, x.shape[2])
        # x_unshuffled.scatter_(dim=1, index=indices, src=x_shuffled)

    def forward_encoder(self, x, target):
        # embed patches
        x = self.patch_embed(x)
        B, N, C = x.shape
        h = w = int(np.sqrt(N))
        pos_embed = self.pos_embed.clone().reshape(1, h, w, C).repeat(B,1,1,1)
        decoder_pos_embed = self.decoder_pos_embed.clone().reshape(1, h, w, -1).repeat(B, 1, 1, 1)
        x, pos_embed, decoder_pos_embed, target = self.raster(x.reshape(B, h, w, C), pos_embed, decoder_pos_embed, target)
        x = x + pos_embed[:, :-1, :]

        b, num_seg, seg_size, c = x.shape
        x = x.reshape(b, num_seg*seg_size, c)
        # apply Transformer blocks
        features = []
        count = 0
        for blk in self.blocks:
            x = blk(x, self.mask)
            count += 1
            if count == 6 or count == 8 or count == 10 or count == 12:
                features.append(x)
        # x = self.norm(x)
        ar_features = []
        kd_features = []
        count=0
        for norm in self.ar_enc_norm:
            ar_features.append(norm(features[count]))
            count+=1
        count = 0
        for norm in self.kd_enc_norm:
            kd_features.append(norm(features[count]))
            count += 1
        x_kd = self.kd_enc2dec(torch.cat(kd_features, dim=-1)).reshape(b, num_seg, seg_size, self.decoder_embed_dim, self.decoder_depth)
        x_ar = self.ar_enc2dec(torch.cat(ar_features, dim=-1)).reshape(b, num_seg, seg_size, self.decoder_embed_dim, self.decoder_depth)

        return x_kd, x_ar, pos_embed, decoder_pos_embed, target

    def forward_decoder(self, latent_ar, latent_kd, decoder_pos_embed):
        # embed tokens
        b, num_seg, seg_size, c, _ = latent_ar.shape
        ar_token = self.ar_token.unsqueeze(1).repeat(b, num_seg, seg_size, 1)+ decoder_pos_embed[:, 1:]
        ar_token = ar_token.reshape(b, num_seg * seg_size, -1)
        kd_token = self.kd_token.unsqueeze(1).repeat(b, num_seg, seg_size, 1) + decoder_pos_embed[:, :-1]
        kd_token = kd_token.reshape(b, num_seg * seg_size, -1)
        # add pos embed
        latent_ar = latent_ar.reshape(b, num_seg*seg_size, self.decoder_embed_dim, self.decoder_depth)
        latent_kd = latent_kd.reshape(b, num_seg * seg_size, self.decoder_embed_dim, self.decoder_depth)

        # apply Transformer blocks
        count = 0
        for blk in self.ar_blocks:
            ar_token = blk(ar_token, latent_ar[:, :, :, count], self.mask)
            count += 1
        ar_token = self.ar_norm(ar_token)
        ar_token = self.ar_pred(ar_token)
        ar_token = ar_token.reshape(b, num_seg, seg_size, -1)

        count = 0
        for blk in self.kd_blocks:
            kd_token = blk(kd_token, latent_kd[:, :, :, count], self.mask)
            count += 1
        kd_token = self.kd_norm(kd_token)
        kd_token = self.kd_pred(kd_token)
        kd_token = kd_token.reshape(b, num_seg, seg_size, -1)
        return ar_token, kd_token

    def forward_loss(self, pred, teacher_out):
        pred = pred / pred.norm(dim=-1, keepdim=True)
        teacher_out = teacher_out / teacher_out.norm(dim=-1, keepdim=True)
        assert pred.shape == teacher_out.shape
        loss = 2 - 2 * (pred * teacher_out).sum(dim=-1)
        return loss

    def forward(self, imgs, target):
        B,N,C=target.shape
        target = target[:, 1:].reshape(B, 14,14,C)
        latent_ar, latent_kd, pos_embed, decoder_pos_embed, target = self.forward_encoder(imgs, target)
        ar_pred, kd_pred = self.forward_decoder(latent_ar, latent_kd, decoder_pos_embed)  # [N, L, p*p*3]
        ar_loss = self.forward_loss(ar_pred, target[:, 1:])
        kd_loss = self.forward_loss(kd_pred, target[:, :-1])
        return ar_loss.mean(0).mean(1), kd_loss.mean(0).mean(1)


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
