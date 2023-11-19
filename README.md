# Using convolutional neural network to enhance the image quality of 5.8G analog VTX
使用卷积神经网络用于加强5.8G模拟图传（常见于航模）的画面，减轻信号干扰噪声对视觉的影响

# 训练集
使用监督学习对网络进行训练，训练集包括一组有干扰噪声的图像和一组没有干扰噪声并且图像位置一一配对的图像。分别由地面上的图传接收屏幕和机载的DVR模块（我将一个图传接收器绑在飞机上😂）录制。将两个视频位置和时间对齐并重采样为480*320，使用prepare.ipynb读取视频并取出每帧（或间隔n帧取出）。

使用prepare.ipynb时，修改变量指定视频输入路径、输出路径和帧采样间隔，就可以进行帧提取并自动按视频帧命名。

# 网络结构
图传的噪声主要是条状的偏色，网络需要对其进行修复。借鉴了pix2pix使用的网络，本项目使用了带跨层连接结构的编码器-解码器网络。每层都是Conv-BatchNorm-ReLU结构

![无标题](https://github.com/LoRafyw/deepVRX/assets/138299454/579849ac-0827-4039-a9fb-73d4f19d4198)

判别器使用一个常规的卷积判别器网络，使用多层Conv-BatchNorm-LeakyReLU结构进行特征提取，最后展平送入全连接神经网络进行分类。

# 训练
使用监督监督学习，使用L2损失和对抗损失作为生成器的损失。L2损失用于图像的整体修复，使用对抗损失增强生成图像的清晰度。对抗损失的权重相对较小，因为在训练过程中发现过大的对抗损失权重会给画面带来奇怪的纹理（可能是因为训练集无干扰图片质量不高）。

本项目主要消除图像中彩条状偏色的噪声，生成完整的画面，所以注重对图像整体的感知，纹理修复在本项目中并不是主要目的。因此在本项目中没有使用类似pix2pix的patch-GAN结构进行纹理感知，而是使用一个整体输入的判别器，并且对于480*320的图像并没有分块处理而是整体使用网络输入输出。

在train.ipynb中，通过修改p2pDataset类中的noise_path和clean_path来指定两种图片存放的路径，对应图片的名称要相同（prepare.ipynb的输出正是如此）。开始运行前，先要在系统控制台中输入python -m visdom.server启动visdom服务器，运行时可以通过visdom实时查看。

# 推理
使用eval.ipynb指定输入的视频和输出的路径，就可以进行推理，此过程仅消耗0.5G显存。左到右三张图片分别为：原始有干扰的画面，模型处理增强后的画面，真实无干扰画面

![屏幕截图 2023-10-29 161022](https://github.com/LoRafyw/deepVRX/assets/138299454/c1acaea0-c698-41fa-b538-a4c432419795)

![屏幕截图 2023-10-29 161639](https://github.com/LoRafyw/deepVRX/assets/138299454/ef4cc930-cde3-4168-a923-a869e5370d51)

在香橙派5上使用rknn_eval.py进行实时推理
# 补充
由于时间原因，使用的训练集较少，当前项目中模型泛用性不强，需要后续训练

只能减轻较弱干扰，对于过强干扰（出现雪花）效果较差
