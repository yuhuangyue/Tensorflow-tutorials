# Tensorflow-tutorials
记录于2018.7.26 <br>
一些用tensorflow实现的简单算法<br><br>

##1. Liner Regression & Logistic Regression<br>
https://blog.csdn.net/yunhaitianguang/article/details/43877591
<br><br>

##2. Net<br>
train_op<br>predict_op<br><br>

##3. Autoencoder<br>
自己写的一个autoencoder(05_1)<br>
用卷积神经网络写的<br>
层次是这样的 1（灰度图）-- 32 -- 64 -- 128 -(中间可以提取出一个特征向量)- 128 -- 64 -- 32 --1<br>

06那个教程里面给输入图片加了noise<br>
正规做法是应该这样<br>
保证模型能适应各种图片<br>

**特别注意这里的卷积和解卷积的过程<br>
另外这个里面的vis函数可以借鉴一下<br><br>**

##4. Word2Vec
* skip-gram<br>
* CBOW<br>
https://zhuanlan.zhihu.com/p/27234078


