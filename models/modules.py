import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
        

class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()
        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_lp = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, x_parallel):
        rgb, lp = x_parallel
        rgb = self.se_rgb(rgb)
        lp = self.se_lp(lp)
        return [rgb + lp, lp]


class Concatenate(nn.Module):
    def __init__(self, channels):
        super(Concatenate, self).__init__()
        self.conv = nn.Conv2d(2*channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        rgb_x, lp_x = x[0], x[1]
        result = torch.cat((rgb_x, lp_x), dim=1)
        result = self.conv(result)
        return [result, lp_x]


class Exchange(nn.Module):
    def __init__(self, bn, bn_threshold):
        super(Exchange, self).__init__()
        self.bn = bn
        self.bn_threshold = bn_threshold
    def forward(self, x):
        bn1, bn2 = self.bn[0].weight.abs(), self.bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= self.bn_threshold] = x[0][:, bn1 >= self.bn_threshold]
        x1[:, bn1 < self.bn_threshold] = x[1][:, bn1 < self.bn_threshold]
        x2[:, bn2 >= self.bn_threshold] = x[1][:, bn2 >= self.bn_threshold]
        x2[:, bn2 < self.bn_threshold] = x[0][:, bn2 < self.bn_threshold]
        return [x1, x2]


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        # x.size() = (batch_size, chanenel, height, width)
        # L2Norm
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        #weight.size() = (512) -> (1,512,1,1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ModuleParallel_NonSharing(nn.Module):
    def __init__(self, module, num_parallel=2):
        super(ModuleParallel_NonSharing, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'module_'+str(i), module)

    def forward(self, x_parallel):
        return [getattr(self, 'module_'+str(i))(x) for i, x in enumerate(x_parallel)]
    def get_module(self, i):
        return getattr(self, 'module_'+str(i))

class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        # Batch_size, Num_tokens, Channels
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors,
                 embd_pdrop, attn_pdrop, resid_pdrop):
        """

        Args:
            n_embd: Hidden Dimension for each tokens (should be number channels of current stage)

            n_head: Num head
            block_exp: Scale factor for Feed forward layer
            n_layer: number of Block
            vert_anchors: number of vertical tokens (we will reduce number of tokens into a fixed to prevent quadratic grow
            of transformer)
            horz_anchors: number of horizontal tokens
            embd_pdrop: Embeding dropout
            attn_pdrop: Attention dropout
            resid_pdrop: Feed forward dropout

            config: config file
        """
        super().__init__()
        self.n_embd = n_embd
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # Average Pooling (pool original H,W into fixer vert_anchors, horz_anchors)
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # token projection:
        #self.tok_emb = nn.Linear(n_embd, n_embd)
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 2 * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def forward(self, x):
        """
        Args:
            X:
                image_tensor (tensor): B, C, H, W
                lp_tensor (tensor): B, C, H, W

        """
        image_tensor = x[0]
        lp_tensor = x[1]
        # To B,C,8,8
        H,W = lp_tensor.shape[2:4]
        image_tensor = self.avgpool(image_tensor)
        lp_tensor = self.avgpool(lp_tensor)
        bz = lp_tensor.shape[0]
        c, h, w = lp_tensor.shape[1:4]
        # forward the image model for token embeddings
        # B,C,8,8 -> B,C,64
        image_tensor = image_tensor.view(bz, -1, h*w)
        lp_tensor = lp_tensor.view(bz, -1, h*w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lp_tensor], dim=2).permute(0, 2, 1).contiguous() # B,128,C
        #token_embeddings = self.tok_emb(token_embeddings) # B, 128, n_emb

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings)  # (B, 128, C)
        x = self.blocks(x)  # (B, 128, C)
        x = self.ln_f(x)  # (B, 128, C)
        x = x.view(bz, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, 0, :, :, :].contiguous().view(bz, -1, h, w)
        lp_tensor_out = x[:, 1, :, :, :].contiguous().view(bz, -1, h, w)

        image_tensor_out = F.interpolate(image_tensor_out, size=(H,W), mode="bilinear")
        lp_tensor_out = F.interpolate(lp_tensor_out, size=(H,W), mode="bilinear")

        return [image_tensor_out, lp_tensor_out]

