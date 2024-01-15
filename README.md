# LocalGCN

## Environmental Settings
Refer to the requirements.txt

## Usage
You can run three models with various options.
For more detailed information about the arguments, you can check by running the command:

```bash
cd code
python main.py --help
```

Before running LocalGCN and AnchorEmbRec, make sure to put the pretrained model in the embed folder.
Refer to the example files inside for the format:
{pretrained_model_name}-{dataset_name}.pt

1. LightGCN

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126). (SIGIR '20)

```bash
cd code
python main.py --config_file ../config/lgcn_ml1m.yaml
```

2. LocalGCN

IMPGCN-based model using LOCA sub-community generation method. 
> Fan Liu, Zhiyong Cheng*, Lei Zhu, Zan Gao and Liqiang Nie*. Interest-aware Message-Passing GCN for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2102.10044). (WWW '21) (“*”= Corresponding author)
> Minjin Choi, Yoonki Jeong, Joonseok Lee, Jongwuk Lee. LOCA: Local Collaborative Autoencoders,  [Paper in arXiv](https://arxiv.org/abs/2103.16103) (WSDM '21)

```bash
cd code
python main.py --config_file ../config/localgcn_ml1m.yaml
```

3. AnchorEmbRec

LightGCN-based recommendation using only the anchor user representations created by LOCA sub-community generation method.
>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126). (SIGIR '20)

```bash
cd code
python main.py --config_file ../config/anchorrec_ml1m.yaml
```