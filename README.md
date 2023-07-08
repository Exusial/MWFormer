## MWFormer: Mesh Understanding with Window-Based Transformer

Hao-Yang Peng, Meng-Hao Guo, Zheng-Ning Liu, Yong-Liang Yang, Tai-Jiang Mu

This project contains the raw implementation of MWFormer, which should run on Linux OS with python3.8+ installed.

To install the requirements, run:
```
pip install -r requirements.txt
```

To reproduce the result of segmentation table in rep.png, just run:
```
sh scripts/test_swin_chairs.sh
```
Note that COSEG datasets should be downloaded with 
```
sh scripts/get_data.sh
```

Since the codes are in a raw situation, if you meet any problems in running the code, please contant our email: phy22@mails.tsinghua.edu.cn. We thank for your patient reviewing of the codes.