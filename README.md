输入图在data里面
输出图在output_results/3E/dynamic..../hdr_output里面
test_3E.py是生成输出图文件

  计算机视觉期末作业

  ——HDRFlow模型的训练与测试

  组员：郝泽其、陈诣勋、段小米

  1. 摘要
     
本次实验中，我们小组搜索了很多资料，查阅了很多相关文献，最终主要参考学习了一篇HDRFlow来进行本次实验作业，利用HDRFlow模型来进行HDR图像的生成。在报告中介绍了HDRFlow模型在高动态范围（HDR）图像处理任务中的应用，包括模型的构建和测试过程，因为使用了预训练模型，故没有训练过程，但也会对训练过程进行一定的说明。通过使用合成和真实数据集，对模型进行了系统的评估，重点分析了模型在不同曝光条件下的表现。

HDR图像技术通过捕捉不同曝光时间的图像，生成更高质量、更高动态范围的图像。HDRFlow是一种针对HDR图像生成的深度学习模型。本文档详细描述了HDRFlow模型的测试流程，包含数据准备、模型构建、损失函数设计、训练策略和测试评估等环节。

  2. 数据准备
数据集介绍：
HDRFlow模型使用了多种数据集，包括Vimeo Septuplet数据集、Sintel数据集、动态RGB数据和静态RGB数据。

  3. 模型构建

3.1 模型结构
HDRFlow模型是一个基于深度学习的HDR视频生成模型，主要由卷积神经网络（CNN）和光流估计模块组成。其核心在于通过多个曝光的LDR图像生成高质量的HDR图像，并对图像之间的运动进行准确估计。HDRFlow模型主要由特征提取网络、流估计模块和HDR图像生成模块组成。模型文件主要包括model_3E.py和utils.py。
3.1.1 特征提取网络
HDRFlow类在model_3E.py中定义，该类通过卷积神经网络提取图像特征，并通过光流估计模块计算不同曝光时间之间的像素位移。
3.1.2 损失函数
HDRFlow_Loss_3E类在loss.py中定义，用于计算模型的损失，包括光流损失和HDR重建损失。光流损失确保模型能够准确估计不同曝光时间之间的像素位移，HDR重建损失则确保生成的HDR图像质量。
损失函数文件 loss.py 定义了用于训练HDRFlow模型的损失函数，包括HDR图像生成损失和光流估计损失
3.1.3 辅助功能
utils.py提供了一系列辅助功能，如图像的预处理和后处理、损失计算和评估指标等。这些函数在训练和测试过程中被频繁调用。

  4. 模型训练
因为已经提供了预训练模型，所以这里只是介绍一下训练文件及相关设置。train文件是直接使用的开源代码。

4.1 训练参数设置
通过 argparse 解析命令行参数，设置训练超参数和路径信息。
训练参数设置如下：
学习率：0.0001
batch size：16
训练轮数：40
学习率衰减：每20轮衰减一次
训练脚本train_3E.py负责HDRFlow模型的训练。该脚本首先通过argparse库解析命令行参数，包括数据集路径、日志保存路径、学习率、批量大小和训练轮数等。

4.1.1 参数解析
get_args函数定义了所有训练相关的参数，如数据集路径、学习率、批量大小和训练轮数等，并通过命令行传递这些参数。

4.2 数据加载
数据预处理与加载是模型训练和评估的重要步骤。我们通过 dataset.py 文件定义了数据集的加载与处理方法。
数据加载和预处理使用了PyTorch的DataLoader模块，通过fetch_dataloader函数实现数据的加载和预处理。该函数根据给定的路径和批量大小返回DataLoader对象，用于训练过程中的数据迭代。数据加载器 fetch_dataloader 用于加载Vimeo Septuplet和Sintel数据集。

4.3 训练过程
模型训练过程如下：
初始化模型和优化器。
迭代训练数据，计算损失并更新模型参数。
每隔一定步数记录训练状态和损失值。
训练过程由train函数实现，该函数进行模型的前向传播、损失计算、反向传播和参数更新。每个epoch结束后，训练脚本会调用validation函数进行模型验证。

4.3.1训练循环
训练函数 train 定义了每个训练周期的具体步骤，包括数据加载、前向传播、计算损失和反向传播。

4.3.2 前向传播
在每个训练步骤中，模型将输入的低动态范围（LDR）图像和曝光时间送入网络，得到预测的HDR图像和光流估计。

4.3.3 损失计算与反向传播
通过计算预测的HDR图像和光流估计的损失，模型进行反向传播并更新参数。optimizer用于执行参数更新，scheduler用于调整学习率。

4.3.3 日志记录
训练过程中，脚本会定期记录训练损失和耗时，并输出到控制台。训练结束后，模型和优化器的状态会被保存到指定的日志目录中。

4.4 验证过程
每轮训练结束后，使用验证集评估模型的性能，验证函数 validation 用于在验证集上评估模型性能，计算PSNR和SSIM等指标。

  5. 模型测试

5.1 测试参数设置
测试脚本test_3E.py负责HDRFlow模型的测试。该脚本解析命令行参数，包括数据集路径、预训练模型路径、结果保存路径等。
测试参数设置如下：
预训练模型路径：./pretrained_models/3E/checkpoint.pth
测试数据集：DeepHDRVideo 和 CinematicVideo
是否保存结果：True
结果保存路径：./output_results/3E

5.2 数据加载
测试数据由Real_Benchmark_Dataset和Syn_Test_Dataset类加载，这些类负责读取和处理测试图像，确保其适用于模型输入。

5.3 测试过程
测试过程由main函数实现，该函数加载预训练模型，并在测试数据集上进行推理，生成HDR图像。计算PSNR和SSIM等评估指标。

5.3.1 评估指标计算
通过skimage库计算PSNR和SSIM指标，评估模型在不同曝光条件下的表现。指标计算结果会输出到控制台，并保存到指定目录。

5.3.2 结果保存
将生成的HDR图像和光流结果保存到指定目录，并对结果进行分析，比较不同曝光条件下模型的表现。

  6. 实验结果与分析

6.1 训练过程
通过在Vimeo Septuplet和Sintel数据集上训练HDRFlow模型，记录了训练损失随时间的变化。结果表明，模型的训练损失逐渐下降，说明模型逐步学会了如何生成高质量的HDR图像。

6.2 测试结果
在DeepHDRVideo和CinematicVideo数据集上测试模型，评估了模型在不同曝光条件下的表现。结果表明，HDRFlow模型在低、中、高曝光条件下均能生成高质量的HDR图像。训练过程中，模型的损失值逐渐下降，验证集上的PSNR和SSIM指标逐步提升，表明模型在训练数据上的拟合效果良好。

6.2.1 低曝光条件
在低曝光条件下，模型能够准确恢复图像细节，PSNR和SSIM指标较高。

6.2.2 中等曝光条件
在中等曝光条件下，模型表现出色，生成的HDR图像质量高，且PSNR和SSIM指标优异。

6.2.3 高曝光条件
在高曝光条件下，模型也能较好地恢复图像细节，虽然PSNR和SSIM指标稍低，但仍能生成令人满意的HDR图像。

6.3 结果分析
通过分析不同曝光条件下的结果，可以发现HDRFlow模型对不同曝光时间的图像均能较好地处理，说明其在HDR图像生成任务中具有较强的泛化能力。模型在高曝光条件下的表现略逊于低曝光和中等曝光条件，可能是由于高曝光图像的信息损失较大，模型恢复难度较高。

  7. 结论

通过在多种数据集上的训练和测试，验证了HDRFlow模型在HDR图像生成任务中的有效性。实验结果表明，HDRFlow模型能够在不同曝光条件下生成高质量的HDR图像，具有较强的泛化能力。通过结合光流估计和HDR成像技术，实现了对高质量HDR图像的生成。在Vimeo-90K、Sintel、DeepHDRVideo和CinematicVideo数据集上的实验结果验证了模型的有效性。
对部分代码文件的详解

1.network_utils.py
主要包含了用于构建神经网络的工具函数和模块。这些函数大部分与计算机视觉任务中的卷积神经网络 (CNN) 相关，特别是在处理HDR图像和其他图像处理任务时常用的功能。
主要功能和结构概述：
模块导入和网络加载：

import_network(net_name, arch_dir='models.archs', backup_module='networks')：根据给定的网络名称从指定目录中导入网络模型。如果找不到指定的网络模型文件，将回退到备用模块。

激活函数：支持多种激活函数，如LeakyReLU、ReLU、Sigmoid和Tanh。

卷积层：conv_layer()、反卷积层：deconv_layer()、上采样层：upconv_layer()和最近邻上采样层：up_nearest_layer()，每种层次支持添加批归一化层和指定激活函数。

output_conv()：用于输出层的卷积。

fc_layer()：全连接层。

网络块构建：

down_block()和up_block()：用于构建下采样和上采样的网络块，支持不同的下采样和上采样模式。

make_layer()：用于构建ResNet风格的多层网络块。

BasicBlock类和conv3x3()函数：用于构建基本的ResNet块。

特定任务处理：

merge_feats()和merge_hdr()：用于融合特征图和HDR图像。

MergeHDRModule类：实现了将多个HDR图像按权重融合的模块。

coords_grid()、backward_warp()和affine_warp()：用于图像坐标网格生成和仿射变换。

图像处理函数：

ldr_to_hdr()和adj_expo_ldr_to_ldr()：用于将LDR图像转换为HDR图像和调整曝光。

图像尺寸调整：

resize_flow()、resize_img_to_factor_of_k()和pad_img_to_factor_of_k()：用于调整流场和图像尺寸。

这个文件提供了一系列用于构建和处理神经网络的工具函数和模块，特别适用于计算机视觉和图像处理任务。它支持从简单的卷积层到复杂的ResNet结构的构建，并提供了处理HDR图像、仿射变换以及流场调整等高级功能的实现。

2. model_3E.py

构建了一个名为HDRFlow的神经网络模型，该模型用于处理HDR图像合成和流场预测。以下是对该文件的详细解析，结合了之前给出的network_utils.py文件的工具函数和模块。

模块导入和依赖：
导入了math和time等标准库，以及torch和torch.nn用于神经网络构建。
从其他自定义模块中导入了network_utils中的工具函数和类，如激活函数、卷积层、HDR处理函数等。
还从flow_3E和fusion_3E模块分别导入了Flow_Net和Fusion_Net类。

辅助函数：
cur_tone_perturb(cur, test_mode, d=0.7)：根据测试模式生成或保持当前图像的色调扰动，以增强训练的鲁棒性。

准备融合输入数据：
prepare_fusion_inputs(ldrs, pt_c, expos, flow_preds)：根据输入的多个低动态范围图像（LDRs）、曝光时间和流场预测结果，准备用于融合网络输入的数据。

HDRFlow类：
HDRFlow类继承自nn.Module，包含了整个HDR图像合成和流场预测的流程。

__init__方法初始化了两个子模块：

flow_net：使用Flow_Net类实例化的流场预测网络。

fusion_net：使用Fusion_Net类实例化的HDR图像融合网络，输入通道数为54，输出通道数为9，中间层通道数为256。

forward方法定义了前向传播过程：
首先对当前帧图像进行色调扰动处理。
使用flow_net预测输入LDR序列的流场。
调用prepare_fusion_inputs准备融合网络的输入数据。
将融合网络的输入数据和HDR图像传递给fusion_net进行融合操作，得到预测的HDR图像和流场预测结果。
定义了HDRFlow模型的结构，包括特征提取网络、流估计模块和HDR图像生成模块。

3. loss.py

定义了损失函数类：HDRFlow_Loss_3E。用于计算HDR图像合成模型中的损失函数，包括重建损失、HDR对齐损失和光流损失。

模块导入和依赖：
导入了math和numpy等标准库，以及torch和torch.nn用于神经网络构建和损失函数计算。
从network_utils模块中导入了backward_warp函数，用于图像的反向变换。

辅助函数：
tonemap(hdr_img, mu=5000)：对输入的HDR图像进行色调映射处理。

损失函数类：
HDRFlow_Loss_3E类继承自nn.Module，用于计算损失函数。
__init__方法初始化了损失函数的参数，如色调映射系数mu。

forward方法定义了损失函数的计算过程：
pred是模型预测的HDR图像，hdrs是真实的HDR图像列表，flow_preds是光流预测结果，cur_ldr是当前的LDR图像，flow_mask和flow_gts是光流的掩码和真实值。
计算重建损失recon_loss，使用nn.L1Loss()计算预测HDR图像和真实HDR图像的L1损失。
根据当前LDR图像cur_ldr计算掩码mask，用于选择感兴趣的区域。

如果掩码中有非零元素，则计算HDR对齐损失：
使用backward_warp函数将各个HDR图像根据光流预测进行反向变换。
对变换后的HDR图像应用色调映射并计算L1损失。

计算光流损失：
根据光流掩码flow_mask选择感兴趣的区域。
使用nn.L1Loss()计算光流预测和真实值的L1损失，并加权累加到总损失中。
返回最终的总损失值。
loss.py文件定义了两个损失函数类，用于HDR图像合成模型中的损失函数计算。这些损失函数包括重建损失、HDR对齐损失和光流损失。

4. __init__.py

用于将一个目录视为Python包。它包含了两个函数fetch_dataloader_2E和fetch_dataloader_3E，这些函数用于获取用于训练和验证的数据加载器。

模块导入和依赖：
导入了必要的Python库，包括torch和os。
导入了从torch.utils.data模块中的DataLoader和ConcatDataset类。
导入了自定义的数据集类Syn_Vimeo_Dataset和Syn_Sintel_Dataset。

fetch_dataloader_3E函数：
参数和功能与fetch_dataloader_2E函数类似，区别在于：
使用了帧数为5，曝光次数为3的设置来加载数据集。
创建了相应的训练数据加载器train_loader和验证数据加载器val_loader。
__init__.py文件通过函数fetch_dataloader_3E提供了获取训练和验证数据加载器的功能。这些函数利用了Syn_Vimeo_Dataset和Syn_Sintel_Dataset等自定义数据集类来加载不同数据集的图像和HDR数据，并将它们合并为单个数据集对象以进行训练。返回的数据加载器可以直接用于训练深度学习模型，支持批量处理、数据随机打乱和多线程加载等功能。

5. utils.py
提供了一系列辅助功能，如图像的预处理和后处理、损失计算和评估指标等。文件包含了多个实用函数和类，用于图像处理、数据加载和模型训练中的各种功能

6. train_3E.py

HDRFlow模型的训练脚本，负责模型的训练过程，包括前向传播、损失计算、反向传播和参数更新。

导入必要的库和设置环境变量：
使用 argparse 解析命令行参数，可以方便地指定数据集路径、日志目录、学习率等参数。
使用 torch 和 torchvision 库进行深度学习模型的构建和训练。

获取命令行参数：
get_args() 函数定义了所有的命令行参数，例如数据集路径、日志目录、训练超参数等。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/3af24a93-0ebc-4128-bdf9-0942c9144460)

模型训练函数：
train() 函数定义了模型的训练过程。在每个训练循环中，加载数据并将其传输到 GPU 上，计算损失并进行反向传播优化模型参数。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/669089c1-147a-4a44-bd29-7a5ec4165a6f)

模型验证函数：
validation() 函数用于评估模型在验证集上的性能。计算预测图像与真实图像之间的 PSNR（峰值信噪比），并保存最佳模型。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/6f19bfc5-5d82-4b67-aa71-892ad8a197e4)

主函数：
main() 函数是整个训练流程的入口。它初始化模型、损失函数和优化器，并循环执行多个 epoch 的训练和验证过程。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/c5c902d9-850e-43f5-95b3-1862086ce407)

日志和模型保存：
在训练过程中，会保存每个 epoch 结束后的模型参数、优化器状态和训练日志。这些信息保存在指定的日志目录下，便于后续分析和恢复训练。
通过这份文件，可以实现对 HDRFlow 模型的端到端训练过程，包括数据加载、模型构建、损失计算、优化器设置、模型保存等功能，是一个完整的深度学习训练脚本

7. test_3E.py

HDRFlow模型的测试脚本，负责在指定数据集上评估预训练模型的效果，并保存评估结果。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/a40a39d0-9976-40dd-80b7-b0835dab90ce)

使用 argparse 解析命令行参数，支持选择不同的数据集和设置预训练模型路径、保存结果的目录等。
使用 torch 和 torchvision 库进行深度学习模型的加载和推理。
提供了 --dataset 和 --dataset_dir 参数，用于指定数据集类型和数据集路径。
--pretrained_model 指定了预训练模型的路径。
--save_results 选择是否保存评估结果，--save_dir 指定结果保存的目录。
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/5b7103ac-949b-4181-a99e-5b4c80c638b9)
![image](https://github.com/OUC-CV/final-project-ouc_hao/assets/111734393/4d865ddf-51dc-4e82-9600-6d591bf88f2a)

模型加载和设置：
加载 HDRFlow 模型并放置在 GPU 上进行评估。
使用 nn.DataParallel 支持在多 GPU 上进行测试。

数据集加载：
根据 --dataset 参数选择加载不同的测试数据集，包括 DeepHDRVideo 和 CinematicVideo。
使用 DataLoader 加载数据集，并设置适当的 batch size 和数据处理参数。

结果保存：
如果选择保存结果 (--save_results 为 True)，则将评估结果保存为图像文件，并根据曝光强度将结果分类保存。

输出结果：
在测试过程中，输出每个图像的评估结果，包括 PSNR 和 SSIM 的平均值以及每个曝光强度段的平均值。
通过这份文件，可以实现对 HDRFlow 模型在不同数据集上的性能评估和结果保存，帮助分析模型在真实场景和合成场景下的表现差异，是进行模型效果验证和比较的重要工具。
以上代码文件的协同工作，实现了HDRFlow模型的HDR图像生成任务。
