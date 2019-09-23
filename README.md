# gesture-vrep
通过摄像头进行手势识别，控制vrep仿真机械臂
anaconda 环境部署：
conda create -n py37 python=3.7
pip install opencv-python
conda install tensorflow==1.13.1
conda install keras==2.2.4
conda install matplotlib
pip install imutils

程序介绍：
label.py   采集，制作标签数据集
model.py   定义模型
train.py   模型训练
test.py    测试手势识别结果
main.py    控制机械臂主程序
vrep.py, vrepConst.py, remoteApi.so : vrep 为 windows 提供的官方接口
puma560.ttt   vrep机械臂

模型使用步骤：
 1. python label.py  按照label,采集数据集，因为使用环境不同，所以最好重新采集数据集。
 			按 ‘c' 键，采集当前镜头框图片，作为一个label，大概按50次，采集50张。
			按 ’n' 键，采集下一个label的图片，换一个手势，50张。
			直到所有label都采集完
			按’q‘键，退出程序，程序会自动将数据集分为训练集和验证集。
			注意： 1. label 'n' 是背景图片，不用比划手势。
				手势可以自定，但最好每个手势差异大一些。

2. python train.py   直接运行，设定的epoch 为 200 次，运行结束自动保存模型 'model/class.model'.

3. vrep中打开机械臂模型，点击运行。
   python main.py    主程序，注意此时的手势，要和采集数据集时相似，这样模型才能预测正确。
