from PIL import Image
import paddle
import numpy as np

paddle.set_device('cpu')

def main():
    # t= paddle.zeros([3,3])
    # print(t)

    # t=paddle.randn([3,3])
    # print(t)

    # img = np.array(Image.open('./1.jpg'))
    # print(img.shape)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         print(f'{img[i,j]:03} ',end='')
    #     print()

    # t=paddle.to_tensor(img,dtype='float32')
    # print(type(t))#img类型
    # print(t.dtype)#内部数据类型

    # print(t.transpose([1,0]))

    t=paddle.randint(0,10,[5,15])
    qkv = t.chunk(3,-1)#切3个 -1维度 最后一个15/3=5
    # print(type(qkv))
    # print(qkv)
    q,k,v=qkv
    print(q)

if __name__ == "__main__":
    main()
