代码组织和运行逻辑：
00. funcs.py
    定义一些便于其它程序调用的函数

01. get_supporting_arrays.py
    生成所需要的k空间和x空间的球谐函数array，结果默认储存在程序所在路径下Y_lm_arrays文件夹内
    （如果只需要测量monopole，不需要运行该程序，注意给足足够的硬盘存储空间）

02. Full_VT.py
    生成所有可能构型的VT，导出VT_FFT.txt默认储存在程序所在路径下。

03. mass_interpolation.py
    将各种Weight分配到mesh上，输出FKP_field和N_field以及各种元数据，存储在指定的文件夹下。
    特别注意如果采用批量测量多个realization，留足硬盘空间。
    基于mpi4py，多进程加速。注意将进程数设置为2,5,7等质数

04. B_mono_full.py
    读取mass_interpolation.py生成的FKP_field和N_field，各种元数据以及VT_FFT.txt，计算B_monopole, 
    输出结果到Results文件夹。



