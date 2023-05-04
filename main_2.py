import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image
paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class Mlp(nn.Layer):
    def __init__(self,embed_dim,mlp_ratio=4.0,dropout=0.):
        super().__init__()
        self.fc1=nn.Linear(embed_dim,int(embed_dim *mlp_ratio))
        self.fc2 =nn.Linear(int(embed_dim *mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()
    
    
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.dropout(x)
        return x
#图->分块->映射（可学习）->特征
class PatchEmbedding(nn.Layer):
    def __init__(self,image_size,patch_size,in_channels,embed_dim,dropout=0.):
        super().__init__()
        self.patch_embed = nn.Conv2D(in_channels,
        embed_dim,
        kernel_size= patch_size,
        stride=patch_size,
        weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
        bias_attr=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        # print('进入',x.shape)
        #x:[1,1,28,28]
        x=self.patch_embed(x)
        # x:[n,embed_dim,h',w']
        x=x.flatten(2)# n ,embed_dim,h *w'
        x= x.transpose([0,2,1])
        x= self.dropout(x)
        # print('进入',x.shape)
        return x
#编码输入 input=x
class Encoder(nn.Layer):
    def __init__(self,embed_dim):
        super().__init__()
        self.attn = Identity()
        self.attn_norm =nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def  forward(self,x):
        print('进1',x.shape)
        h=x
        x=self.attn_norm(x)
        x = self.attn(x)
        x = h+x


        h=x
        x=self.mlp_norm(x)
        x = self.mlp(x)
        x = h+x
        print('进2',x.shape)
        return x        
class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed=PatchEmbedding(224,7,3,16)
        layer_list = [Encoder(16) for i in range(1)]
        self.encoders= nn.LayerList(layer_list)
        self.head=nn.Linear(16,10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self,x):
        # print('进入',x.shape)
        x=self.patch_embed(x)
        for encoder in self.encoders:
            x=encoder(x)

        x= x.transpose([0,2,1])
        x=self.avgpool(x)#[n,c,1]
        x=x.flatten(1)#[n,c]
        x=self.head(x)
        # print('出来',x.shape)
        return x

def main():
    # img=Image.open('./1.jpg')
    # img = np.array(img)
    # for i in range(28):
    #     for j in range(28):
    #         print(f'{img[i:j]:03}',end='')
    #     print()
    
    # sample = paddle.to_tensor(img,dtype='float32')
    #     #simulate a batch of  data
    # sample = sample.reshape([1,1,28,28])

    # print(sample.shape)
    
    # #.2patch_enbeding
    # patch_embed= PatchEmbedding(image_size = 28 ,patch_size =7,in_channels=1,
    #     embed_dim=1)
    # out =patch_embed(sample)
    # for i in range(0,28,7):
    #     for j in range(0,28,7):
    #         print(paddle.sum(sample[0,0,i:i+7,j:j+7]).numpy().item)
    # # print(out.shape)

    # #3mlp
    # mlp=Mlp(1)
    # out=mlp(out)
    # print('out.shape=',out.shape)    
    t= paddle.randn([4,3,224,224])
    model=ViT()
    out=model(t)
    print(out.shape)

if __name__=="__main__":
    main()