# yolov5_tflite_on_ios
- [x] Swift
- [x] tflite

# Usage
- tflite example [here](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios)

# Installation:
- Note : The version of TensorFlowLiteSwift in podfile should match with the version of tensorflow when you convert your model used.
```shell
cd yolov5_tflite_demo_ios
pod install
```  
- Install on Xcode
- Put your `yolov5.tflite` model and `label.txt` in `ObjectDetection/Model` 
- Run