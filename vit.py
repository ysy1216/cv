import torch
import torch.nn as nn
# from torchinfo import summary

class Mlp(nn.Module):
    def __init__(self,embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1  = nn.Linear(embed_dim, int(embed_dim* mlp_ratio))
        self.fc2  = nn.Linear(int(embed_dim* mlp_ratio),embed_dim)
        self.act  = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x

class Attention(nn.Module):
    def __init__(self,embed_dim, num_heads, qkv_bias=False, dropout=0.,attention_dropout=0.):
        super().__init__()
        self.embed_dim =embed_dim
        self.num_heads =num_heads
        self.head_dim = int(embed_dim/num_heads)
        self.all_head_dim = self.head_dim*num_heads
        # 把所有q 写在一起， 所有k、V写在一起，然后拼接起来，前1/3代表了所有head的Q，每一个head的尺寸已经定义好，要用的时候切就行了
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim*3,
                             bias=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim,embed_dim)

    def transpose_multi_head(self,x):
        # x: [B, N, all_head_dim]
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        x = x.permute(0,2,1,3)
        # x: [B, num_heads, num_patches, head_dim]
        
        return x


    def forward(self,x):

        B,N ,_ = x.shape
        qkv = self.qkv(x).chunk(3,-1)
        # [B, N, all_head_dim]* 3 ， map将输入的list中的三部分分别传入function，然后将输出存到q k v中
        q, k, v = map(self.transpose_multi_head,qkv)
        # q,k,v: [B, num_heads, num_patches, head_dim]
        attn = torch.matmul(q,k.transpose(-1,-2))   #q * k'
        attn = self.scale * attn
        attn = self.softmax(attn)
        # dropout
        # attn: [B, num_heads, num_patches, num_patches]

        out = torch.matmul(attn, v)  # 这里softmax(scale*(q*k')) * v
        out = out.permute(0,2,1,3)
        # out: [B,  num_patches,num_heads, head_dim]
        out = out.reshape([B, N, -1])

        out = self.proj(out)

        #dropout
        return out


class EncoderLayer(nn.Module):
    def __init__(self,embed_dim=768, num_heads=4, qkv_bias=True, mlp_ratio=4.0, dropout=0., attention_drop=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim,num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim,mlp_ratio)

    def forward(self,x):
        # PreNorm
        h = x   #residual
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x+h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x+h
        
        return x

class Encoder(nn.Module):
    def __init__(self,embed_dim,depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer()
            layer_list.append(encoder_layer)

        self.layers = nn.ModuleList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        # print('进2',x.shape)
        x = self.norm(x)
        # print('进出',x.shape)
        
        return x
 

class PatchEmbedding(nn.Module):
    def __init__(self,image_size=224, patch_size=16, in_channels=3, embed_dim=768 ,dropout=0.):
        super().__init__()
        n_patches = (image_size//patch_size) * (image_size//patch_size)
        self.patch_embedding = nn.Conv2d(in_channels = in_channels,
                                     out_channels= embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim =embed_dim
        # TODO: add cls token
        self.class_token = nn.Parameter(nn.init.constant_(torch.zeros(1,1,embed_dim,dtype=torch.float32),1.0))
        # TODO: add position embedding
        self.position_embedding = nn.Parameter(nn.init.trunc_normal_(torch.randn(1,n_patches+1,embed_dim,dtype=torch.float32),std=.02))


    def forward(self,x):
        
        class_tokens =self.class_token.expand([x.shape[0], 1, self.embed_dim]) #for batch
        # x: [N, C, H, W]
        
        x = self.patch_embedding(x) # x: [n, embed_dim, h', w']
        x = x.flatten(2)   #[n, embed_dim, h'*w']
        x = x.permute(0, 2, 1)  #[n,  h'*w', embed_dim]
        x = torch.concat([class_tokens, x], axis=1)

        # print('embeding中：',x.shape)

        x = x + self.position_embedding
        x = self.dropout(x)
        
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 mlp_ratip=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_drop=0.,
                 droppath=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size,)
        self.encoder = Encoder(embed_dim,depth)
        self.classifier = nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        # x:[N, C, H, W]
        
        N, C, H, W = x.shape
        x = self.patch_embedding(x)   # [N, embed_dim, h', w']
        
        # x = x.reshape([N,C,-1]) /  x = x.reshape(x.shape(:2)+[-1])
        x = x.flatten(2)   #比reshape更简单  [N, embed_dim, h'*w'=num_patchrs]
        # x = x.permute(0, 2, 1)  # [N, num_patches, embed_dim]
        x = self.encoder(x)  # [N, num_patches,embed_dim]
        
        x = self.classifier(x[:, 0])
        
        return x

def main():
    
    vit = VisionTransformer()
    # print(vit)
    t= torch.randn([4,3,224,224])
    out = vit(t)
    print(out)
    


if __name__ == '__main__':
    main()
 