# LSTM-RPA 
>论文[《基于LSTM-RPA音乐流行趋势预测研究》](http://kns.cnki.net/kcms/detail/11.2127.TP.20210915.1447.010.html)代码,中文引用请参照知网相关格式。
>Paper [*LSTM-RPA: A Simple but Effective Long Sequence Prediction Algorithm for Music Popularity Prediction*](https://arxiv.org/abs/2110.15790) code.Please refer to the relevant format for English quotation
>请准守开源协议使用代码
# 主要思想 Main ideas

针对LSTM在音乐长趋势预测中历史信息衰减的问题，提出改进的LSTM滚动预测模型，该模型在预测阶段将前一次输入与当前预测结果相结合，使得历史信息可以沿预测趋势方向流动，从而缓解模型在长趋势预测中的历史信息衰减。
![RPA算法](https://i.loli.net/2021/11/16/plvIAR1xsUQPwX5.png)

# 数据集

[比赛链接](https://tianchi.aliyun.com/competition/entrance/231531/introduction?spm=5176.12281957.1004.16.38b024481kfoZj)
[冠亚军答辩视频及PPT](https://pan.baidu.com/s/1sllhhQ9?spm=5176.21852664.0.0.1f4c313fWKy5tK) 密码 aqyw  [原帖](https://tianchi.aliyun.com/forum/postDetail?spm=5176.21852664.0.0.281b379cKcaXC8&postId=99)
[数据集](https://blog.csdn.net/u012111465/article/details/82910586)
> 注：初赛分A，B阶段，A阶段是500万数据，B阶段是2000万数据，复赛是平台赛，约2亿左右的数据，无法下载。

# 实验
## 实验环境
1. tensorflow=1.14.0
2. keras=2.4.3
3. modin=0.6.0
无需GPU
## 预测图
![WX20211116-220624.png](https://i.loli.net/2021/11/16/lzV7Wp9sLE654qQ.png)
## 单特征实验

![WX20211116-215133.png](https://i.loli.net/2021/11/16/iqGIo6357FYz1OA.png)

## 多特征实验
![WX20211116-215610.png](https://i.loli.net/2021/11/16/ymKfFkRMaJvHV8Z.png)
