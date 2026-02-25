帮我编写E:\Project\Re\Retouch\script\cleanData.py，需求如下:
<!-- 首先读取E:\Project\Re\Retouch\dataset\FFHQ_dual_process\fail.txt，以第一行为例Whitening_Smoothing	/groupshare/FFHQ/20000/20001.png，每行分为两部分，第一部分是操作项Whitening_Smoothing，第二部分是图片路径，我要求你筛选出所有操作项中包含EyeEnlarging字段的，且路径中倒数第二级名为\20000的所有信息行。 -->


<!-- 读取E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt，以第一行{'Whitening': 90, 'Smoothing': 90, 'FaceLifting': 30, 'EyeEnlarging': 0}	/groupshare/FFHQ_three_process/Whitening_Smoothing_FaceLifting/40000/40000.png为例，每行分为两大部分，第一大部分是操作项字典，分别是操作项与分数组成的键值对，第二部分是图片路径（具体路径有无，知道图片名即可），我要求你筛选出所有操作项中EyeEnlarging所对应的分数值为0的所有信息行，这种信息行代表的是问题图片，请你使用筛选出所有问题图片的图片名，然后去处理E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\Smoothing_FaceLifting_EyeEnlarging下的所有图片数据，筛选出其中所有存在的问题图片。然后将其移动到E:\Project\Re\Retouch\dataset\test\0下，正常的图片移动到E:\Project\Re\Retouch\dataset\total中，在这个过程中如果出现图片同名问题的话，则跳过该图片，但请将同名图片路径写入E:\Project\Re\Retouch\dataset\total\conflict.txt -->


<!-- 处理E:\Project\Re\Retouch\dataset\FFHQ_dual_process\FaceLifting_EyeEnlarging\20000中的数据，要求不包含上一步筛选到的所有图片的同名图片
然后读取E:\Project\Re\Retouch\dataset\FFHQ_settings\excluded_images_list\total.txt中的所有信息，其中每行都代表问题图片的图片名，只不过少写了后缀，后缀可能是png或jpg等，请筛选E:\Project\Re\Retouch\dataset\FFHQ_process\EyeEnlarging_30\00000以及E:\Project\Re\Retouch\dataset\FFHQ_ali_process\EyeEnlarging_30\17000的所有图像数据，要求这两个文件夹内不包含问题图片 -->

<!-- 最终将两个文件夹的图片合并到E:\Project\Re\Retouch\dataset\total中，若有同名图片则跳过处理，但是请将同名文件路径写入E:\Project\Re\Retouch\dataset\total\conflict.txt中。 -->
<!-- 最终重新定义问题图片，请你读取E:\Project\Re\Retouch\dataset\FFHQ_settings\excluded_images_list\total.txt中的所有信息，其中每行都代表问题图片的图片名，只不过少写了后缀，后缀可能是png或jpg等，请筛选E:\Project\Re\Retouch\dataset\total中所有的图像数据，要求不包含问题图片。 -->
读取E:\Project\Re\Retouch\dataset\FFHQ_ali_process\one_process.txt，E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_dual_process\dual_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_megvii_dual_process\dual_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_megvii_dual_process\two_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_megvii_process\one_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\three_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_three_process\three_process.txt
E:\Project\Re\Retouch\dataset\FFHQ_megvii_four_process\four_process.txt，
E:\Project\Re\Retouch\dataset\FFHQ_four_process\four_process.txt，
，以{'Whitening': 90, 'Smoothing': 90, 'FaceLifting': 30, 'EyeEnlarging': 0}	/groupshare/FFHQ_three_process/Whitening_Smoothing_FaceLifting/40000/40000.png为例，每行分为两大部分，第一大部分是操作项字典，分别是操作项与分数组成的键值对，第二部分是图片路径（具体路径有无，知道图片名即可），我要求你筛选出所有操作项中EyeEnlarging所对应的分数值为0的所有信息行，这种信息行代表的是问题图片，请你使用筛选出所有问题图片的图片名，然后去处理E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\Smoothing_FaceLifting_EyeEnlarging下的所有图片数据，筛选出其中所有存在的问题图片，然后将问题图片移动到E:\Project\Re\Retouch\dataset\test\0下，正常的图片移动到E:\Project\Re\Retouch\dataset\total中，在这个过程中如果出现图片同名问题的话，则跳过该图片，但请将同名图片路径写入E:\Project\Re\Retouch\dataset\total\conflict.txt
