# quantization-aware training

## 1. 基于tensorflow1.12.0版本实现imception-v3的quantization-aware training， 转化成tflite模型后测试与pb模型测试精度几乎没有下降，模型大小变为原来的1/4,且根据int8推理。

## 2. 步骤：
### (1)：运行train_inception_v3.py得到包含量化节点的ckpt文件。
### (2)：运行freeze_graph_fake_quantized_eval.py得到pb文件。
### (3)：运行eval_pb_convert_to_tflite.py将pb文件转化成tflite模型。
### (4)：运行test_tflite_model.py测试tflite模型的精度。