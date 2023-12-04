from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vit import PatchEmbed, Block, DecoderBlock
from einops import rearrange
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F

class DiGPT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_size = patch_size
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ar_enc2dec = nn.Linear(4 * embed_dim, decoder_depth * decoder_embed_dim)
        self.kd_enc2dec = nn.Linear(4 * embed_dim, decoder_depth * decoder_embed_dim)
        self.ar_enc_norm = nn.ModuleList([nn.LayerNorm(embed_dim)]*4)
        self.kd_enc_norm = nn.ModuleList([nn.LayerNorm(embed_dim)]*4)
        self.decoder_embed_dim=decoder_embed_dim
        self.decoder_depth = decoder_depth
        # --------------------------------------------------------------------------

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.num_seg = 3
        self.seg_size = 49
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
        self.ar_pred = nn.Linear(decoder_embed_dim, 1024, bias=True) # decoder to patch
        self.kd_norm = norm_layer(decoder_embed_dim)
        self.kd_pred = nn.Linear(decoder_embed_dim, 1024, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
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

    def mask_generate(self, segment, tokens_per_segment):
        mask = torch.tril(torch.ones((segment, segment), dtype=torch.float))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=1)
        mask = F.pad(mask.unsqueeze(0).unsqueeze(0), (0, 1, 0, 1), mode="reflect").squeeze(0).squeeze(0)
        return mask

    def raster(self, x, pos_embedding, decoder_pos_embed, label):
        # x   B h w C
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        label = rearrange(label, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=8, p2=8)
        pos_embedding = rearrange(pos_embedding, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        decoder_pos_embed = rearrange(decoder_pos_embed, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=7, p2=7)
        bz, num_seg, seg_size, c = x.shape
        raster_matrix = torch.stack([torch.randperm(num_seg) for _ in range(bz)], dim=0)
        raster_matrix = raster_matrix[:, :, None, None].repeat(1, 1, seg_size, c).to(x.device)
        x_shuffled = torch.gather(x, dim=1, index=raster_matrix)
        label_shuffled = torch.gather(label, dim=1, index=raster_matrix[:, :, 0:1, 0:1].repeat(1, 1, 64, label.shape[-1]))
        pos_embedding = torch.gather(pos_embedding, dim=1, index=raster_matrix)
        decoder_pos_embed = torch.gather(decoder_pos_embed, dim=1, index=raster_matrix[:, :, :, 0:1].repeat(1, 1, 1, decoder_pos_embed.shape[-1]))
        return x_shuffled[:, :-1], pos_embedding, decoder_pos_embed, label_shuffled

    def forward_encoder(self, x, target):
        # embed patches
        x = self.patch_embed(x)
        B, N, C = x.shape
        h = w = int(np.sqrt(N))
        pos_embed = self.pos_embed[:, 1:].clone().reshape(1, h, w, C).repeat(B,1,1,1)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        decoder_pos_embed = self.decoder_pos_embed[:, 1:].clone().reshape(1, h, w, -1).repeat(B, 1, 1, 1)
        x, pos_embed, decoder_pos_embed, target = self.raster(x.reshape(B, h, w, C), pos_embed, decoder_pos_embed, target)
        x = x + pos_embed[:, :-1, :]

        x = x.reshape(B, self.num_seg*self.seg_size, C)
        x = torch.cat([x, cls_tokens], dim=1)
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
        x_kd = self.kd_enc2dec(torch.cat(kd_features, dim=-1)).reshape(B, -1, self.decoder_embed_dim, self.decoder_depth)
        x_ar = self.ar_enc2dec(torch.cat(ar_features, dim=-1)).reshape(B, -1, self.decoder_embed_dim, self.decoder_depth)

        return x_kd, x_ar, pos_embed, decoder_pos_embed, target

    def forward_decoder(self, latent_ar, latent_kd, decoder_pos_embed):
        # embed tokens
        B, N, C, depth = latent_ar.shape
        ar_token = self.ar_token.unsqueeze(1).repeat(B, self.num_seg, self.seg_size, 1)+ decoder_pos_embed[:, 1:]
        ar_token = ar_token.reshape(B, self.num_seg * self.seg_size, -1)
        ar_cls = self.ar_token + self.decoder_pos_embed[:, :1]
        ar_token = torch.cat([ar_token, ar_cls.repeat(B, 1,1 )], dim=1)
        kd_token = self.kd_token.unsqueeze(1).repeat(B, self.num_seg, self.seg_size, 1) + decoder_pos_embed[:, :-1]
        kd_cls = self.kd_token+self.decoder_pos_embed[:, :1]
        kd_token = kd_token.reshape(B, self.num_seg * self.seg_size, -1)
        kd_token = torch.cat([kd_token, kd_cls.repeat(B, 1,1 )], dim=1)

        # apply Transformer blocks
        count = 0
        for blk in self.ar_blocks:
            ar_token = blk(ar_token, latent_ar[:, :, :, count], self.mask)
            count += 1
        ar_token = self.ar_norm(ar_token)
        ar_token = self.ar_pred(ar_token)

        count = 0
        for blk in self.kd_blocks:
            kd_token = blk(kd_token, latent_kd[:, :, :, count], self.mask)
            count += 1
        kd_token = self.kd_norm(kd_token)
        kd_token = self.kd_pred(kd_token)
        return ar_token, kd_token

    def forward_loss_v1(self, pred, teacher_out):
        # pred B num_seg  Seg_size C
        num_seg = self.num_seg
        seg_size=64
        seg_size_S = 49
        B, C = pred.shape[0], pred.shape[-1]

        pred_rest = F.interpolate(
            pred[:, :-1].reshape(B * num_seg, int(seg_size_S ** 0.5), int(seg_size_S ** 0.5), C).permute(0, 3, 1, 2),
            (int(seg_size ** 0.5), int(seg_size ** 0.5)), mode='bilinear').reshape(B * num_seg, C, seg_size).permute(0, 2, 1).reshape(
            B, num_seg, seg_size, C).reshape(B, num_seg*seg_size, C)
        pred = torch.cat([pred_rest, pred[:, -1:]], dim=1)
        pred = pred / pred.norm(dim=-1, keepdim=True)
        teacher_out = teacher_out / teacher_out.norm(dim=-1, keepdim=True)
        assert pred.shape == teacher_out.shape
        loss = 2 - 2 * (pred * teacher_out).sum(dim=-1)
        return loss

    def forward_loss_v2(self, pred, teacher_out):
        # pred B num_seg  Seg_size C
        pred = pred / pred.norm(dim=-1, keepdim=True)
        teacher_out = teacher_out / teacher_out.norm(dim=-1, keepdim=True)
        assert pred.shape == teacher_out.shape
        loss = 2 - 2 * (pred * teacher_out).sum(dim=-1)
        return loss

    def forward(self, imgs, target):
        B,N,C = target.shape
        new_target = target[:, 1:].reshape(B, 16, 16,C)
        latent_ar, latent_kd, pos_embed, decoder_pos_embed, new_target = self.forward_encoder(imgs, new_target)
        ar_pred, kd_pred = self.forward_decoder(latent_ar, latent_kd, decoder_pos_embed)  # [N, L, p*p*3]

        if self.patch_size==16:
            ar_loss = self.forward_loss_v1(ar_pred, torch.cat([new_target[:, 1:].reshape(B, -1, C), target[:, :1]], dim=1))
            kd_loss = self.forward_loss_v1(kd_pred, torch.cat([new_target[:, :-1].reshape(B, -1, C), target[:, :1]], dim=1))
        elif self.patch_size == 14:
            ar_loss = self.forward_loss_v2(ar_pred, torch.cat([new_target[:, 1:].reshape(B, -1, C), target[:, :1]], dim=1))
            kd_loss = self.forward_loss_v2(kd_pred, torch.cat([new_target[:, :-1].reshape(B, -1, C), target[:, :1]], dim=1))
        return ar_loss.mean(), kd_loss.mean()


def DiGPT_vit_base(**kwargs):
    model = DiGPT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=1024, decoder_depth=2, decoder_num_heads=32,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def DiGPT_vit_large(**kwargs):
    model = DiGPT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=2, decoder_num_heads=32,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

DiGPT_vit_base=DiGPT_vit_base
DiGPT_vit_large=DiGPT_vit_large