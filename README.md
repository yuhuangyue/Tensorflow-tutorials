# Tensorflow-tutorials
记录于2018.7.26 <br>
一些用tensorflow实现的简单算法<br><br>

## 1. Liner Regression & Logistic Regression<br>
https://blog.csdn.net/yunhaitianguang/article/details/43877591
<br><br>

## 2. Net<br>
train_op<br>predict_op<br><br>

## 3. Autoencoder<br>
自己写的一个autoencoder(05_1)<br>
用卷积神经网络写的<br>
层次是这样的 1（灰度图）-- 32 -- 64 -- 128 -(中间可以提取出一个特征向量)- 128 -- 64 -- 32 --1<br>

06那个教程里面给输入图片加了noise<br>
正规做法是应该这样<br>
保证模型能适应各种图片<br>

**特别注意这里的卷积和解卷积的过程<br>
另外这个里面的vis函数可以借鉴一下<br><br>**

## 4. Word2Vec<br>
* skip-gram<br>
* CBOW<br>
https://zhuanlan.zhihu.com/p/27234078

## 5. Tensorboard <br>

* 使用tf.summary.scalar记录标量 
* 使用tf.summary.histogram记录数据的直方图 
* 使用tf.summary.distribution记录数据的分布图 
* 使用tf.summary.image记录图像数据 

(1) 善于使用with tf.name_scope("")对可视化出来的graph进行完善，去除多余的东西<br>
(2) 学会tf.summary.image记录图像数据 <br>
**可以看05_1里面我写的可视化部分**<br><br>

## 6. Save & Restore <br>
介绍了saver.restore()和saver.save()两种方法<br>
**restore:重新恢复模型，如果训练在过程中被中断，可以使用restore重新定位到最新一次训练出来的模型继续跑<br>**

