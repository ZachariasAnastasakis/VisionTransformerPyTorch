# A simple implementation of a Vision Transformer (ViT) from scratch using PyTorch.

This is my simple implementation of the Vision Transformer using PyTorch. The official paper of **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** can be found *[here](https://arxiv.org/abs/2010.11929)*.

My custom ViT is trained and tested on 2 datasets, MNIST and CIFAR10 with Pytorch 1.10.0. You can train the above ViT using the command:
```bash
git clone https://github.com/ZachariasAnastasakis/VisionTransformerPyTorch.git
python3 train_vit.py --img_size=32 --in_channels=3 --emb_dim=768 --patch_size=4 --depth=6 --num_heads=8 --mlp_ratio=4 --epochs=10 --num_classes=10 --dataset=MNIST
```

Afterwards, the trained model will be saved to your current directory. You will be able to test it and view the attention maps using the **inference_vis_attn.ipynb**.

## Results on test set

### MNIST dataset
<img src="https://raw.githubusercontent.com/ZachariasAnastasakis/VisionTransformerPyTorch/main/images/mn.png?token=GHSAT0AAAAAACDQTSIH66EYVG47QZZSEROGZECGW3Q" alt="drawing" width="250"/>
<img src="https://raw.githubusercontent.com/ZachariasAnastasakis/VisionTransformerPyTorch/main/images/mnist_4.png?token=GHSAT0AAAAAACDQTSIHEBWT2OMKQH33CSEQZECGXQQ" alt="drawing" width="250"/>
<img src="https://raw.githubusercontent.com/ZachariasAnastasakis/VisionTransformerPyTorch/main/images/mnist_7.png?token=GHSAT0AAAAAACDQTSIHG7LF7W7BYJWTBXGGZECGYDA" alt="drawing" width="250"/>

### CIFAR10 dataset
![](https://raw.githubusercontent.com/ZachariasAnastasakis/VisionTransformerPyTorch/main/images/cifar10_multiple.png?token=GHSAT0AAAAAACDQTSIHGOP5BO2W4UXOHJ5IZECGYUA)
![](https://raw.githubusercontent.com/ZachariasAnastasakis/VisionTransformerPyTorch/main/images/cifar10_multiple_2.png?token=GHSAT0AAAAAACDQTSIGS3MTLPJZHGRTXFXIZECGZAQ)

The above models are trained only for 10 epochs. Adding more epochs perhaps could lead to better results and thus better attentions.

*You do not understand it if you can not implement it !!*