// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate


struct Recognition {
    let confidence: [Float]
    let className: [Float]
    let rect: [Float]
}
/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum MobileNetSSD {
  static let modelInfo: FileInfo = (name: "yolov5s-fp16-2", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "coco", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  let threadCountLimit = 10

  let threshold: Float = 0.55
  let nmsthreshold: Float = 0.7

  // MARK: Model parameters
  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 320
  let inputHeight = 320

  // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 0.0
    let imageStd:  Float = 255.0

  // MARK: Private properties
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3
  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 4) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }


    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()

    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
    let out1 = pow((Double(inputWidth) / 32.0), 2.0)
    let out2 = pow((Double(inputWidth) / 16.0), 2.0)
    let out3 = pow((Double(inputWidth) / 8.0), 2.0)
    let output_box = Int((out1 + out2 + out3)*3.0)
//    var outbuf = [[[Float]]]()
//    let outbuf = Array(repeating: Array(repeating: Array(repeating: 0, count: 1), count: output_box), count: labels.count + 5)
//    var outputMap = [
//        0 : outbuf
//    ]


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }
    let interval: TimeInterval
    let outi: Tensor
//    let outputBoundingBox: Tensor
//    let outputClasses: Tensor
//    let outputScores: Tensor
//    let outputCount: Tensor
    var yolox: Float = 0.0
    var yoloy: Float = 0.0
    var yolow: Float = 0.0
    var yoloh: Float = 0.0
    var rect: [Float]=[0.0]
    var conf: [Float]=[0.0]
    var classindex: [Float]=[0.0]
    var detections: Array<Recognition>=[]
    var recognitions: Array<Recognition>=[]
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
        
//        try interpreter.allocateTensors()

      // Copy the RGB data to the input `Tensor`.
        try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000
        outi = try interpreter.output(at: 0)
        var out: [Float]
        out = [Float](unsafeData: outi.data) ?? []
//        print(out.count)
//        print(out)
        for i in 0..<output_box {
            var maxClass: Float = 0.0
            var detectedClass: Int = -1
            var classes = Array<Float>(repeating: 0, count: labels.count)
//            print(labels.count)
            let confidence:Float = out[4+i*(5+labels.count)]
            for c in 0..<labels.count {
                classes[c] = out[5+c+i*(5+labels.count)]
            }
//            print(classes)
            for c in 0..<labels.count {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
//            print(confidence * maxClass)
            if (confidence * maxClass > threshold) {
                yolox = out[0+i*(5+labels.count)]*Float(inputWidth)
                yoloy = out[1+i*(5+labels.count)]*Float(inputWidth)
                yolow = out[2+i*(5+labels.count)]*Float(inputWidth)
                yoloh = out[3+i*(5+labels.count)]*Float(inputWidth)
                rect = [(yolox-yolow/2)/Float(inputWidth), (yoloy-yoloh/2)/Float(inputHeight), yolow/Float(inputWidth), yoloh/Float(inputHeight)]
                conf = [confidence * maxClass]
                classindex = [Float(detectedClass)]
                print(confidence * maxClass)
                print(rect)
                print(labels[detectedClass])
                detections.append(Recognition(confidence: conf, className: classindex, rect: rect))
            }
        }
        
        recognitions = nms(list: detections)
//        for r in recognitions {
//            print(r.className)
//        }

//      outputBoundingBox = try interpreter.output(at: 0)
//      outputClasses = try interpreter.output(at: 0)
//      outputScores = try interpreter.output(at: 0)
//      outputCount = try interpreter.output(at: 0)
//        print(outputCount)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Formats the results
    let resultArray = formatResults(
        results: recognitions,
        outputCount: recognitions.count,
      width: CGFloat(imageWidth),
      height: CGFloat(imageHeight)
    )
//    print(resultArray)
//    let resultArray = formatResults(
//      boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
//      outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
//      outputScores: [Float](unsafeData: outputScores.data) ?? [],
//        outputCount: Int(Float(([Float](unsafeData: outputCount.data) ?? [0])[0])),
//      width: CGFloat(imageWidth),
//      height: CGFloat(imageHeight)
//    )
//    print(resultArray)


    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: resultArray)
    return result
  }

    
  func nms(list: Array<Recognition>) -> Array<Recognition> {
    var nmsList: Array<Recognition>=[]
    
    for k in 0..<labels.count {
        var sameClassList: Array<Recognition>=[]
        for i in 0..<list.count {
            if (Int(list[i].className[0]) == k){
                sameClassList.append(list[i])
            }
        }
        
        while (sameClassList.count > 0) {
//            print(sameClassList)
            sameClassList.sort { (first, second) -> Bool in
              return first.confidence[0] > second.confidence[0]
            }
            let detections: Array<Recognition> = sameClassList
            let max: Recognition = detections[0]
            nmsList.append(max)
            sameClassList.removeAll()
//            print(sameClassList)
            
            for j in 1..<detections.count {
                let detection: Recognition = detections[j]
                let ar = xywh2Rect(a: max.rect)
                let br = xywh2Rect(a: detection.rect)
                
//                print(box_iou(a: ar, b: br))
//                print(ar)
                if (box_iou(a: ar, b: br) < nmsthreshold) {
                    sameClassList.append(detection)
                }
            }
        }
//        print(sameClassList)
        
    }
//    print(nmsList.count)
    return nmsList
  }
    func xywh2Rect(a: [Float]) -> [Float] {
        let left = max(0.0, a[0]*Float(inputWidth))
        let top = max(0.0, a[1]*Float(inputHeight))
        let right = min(Float(inputWidth-1), (a[0]+a[2])*Float(inputWidth))
        let bottom = min(Float(inputHeight-1), (a[1]+a[3])*Float(inputHeight))
        let b = [left, top, right, bottom]
        return b
    }
    
    func box_iou(a: [Float], b: [Float]) -> Float {
        return box_intersection(a: a, b: b) / box_union(a: a, b: b)
    }
    
    func box_intersection(a: [Float], b: [Float]) -> Float {
        let w = overlap(x1: (a[0]+a[2]) / 2, w1: a[2]-a[0], x2: (b[0]+b[2]) / 2, w2: b[2]-b[0])
        let h = overlap(x1: (a[1]+a[3]) / 2, w1: a[3]-a[1], x2: (b[1]+b[3]) / 2, w2: b[3]-b[1])
        if (w < 0 || h < 0) {
            return 0
        }
        let area = w * h
        return area
    }
    
  func box_union(a: [Float], b: [Float]) -> Float {
    let i = box_intersection(a: a, b: b)
    let u = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - i
    return u
  }
  
  func overlap(x1: Float, w1: Float, x2: Float, w2: Float) -> Float {
    let l1 = x1 - w1 / 2
    let l2 = x2 - w2 / 2
    let left = l1 > l2 ? l1 : l2
    let r1 = x1 + w1 / 2
    let r2 = x2 + w2 / 2
    let right = r1 < r2 ? r1 : r2
    return right - left
  }

  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
    func formatResults(results: Array<Recognition>, outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
    var resultsArray: [Inference] = []
//    print(outputScores[0])
    if (outputCount == 0) {
      return resultsArray
    }
    for i in 0..<outputCount {
//        print(i)

        let score = results[i].confidence[0]

      // Filters results with confidence < threshold.
      guard score >= threshold else {
        continue
      }

      // Gets the output class names for detected classes from labels list.
        let outputClassIndex = Int(results[i].className[0])
      let outputClass = labels[outputClassIndex]

      var rect: CGRect = CGRect.zero
        
        let boundingBox = results[i].rect

      // Translates the detected bounding box to CGRect.
      rect.origin.x = CGFloat(boundingBox[0])
      rect.origin.y = CGFloat(boundingBox[1])
      rect.size.width = CGFloat(boundingBox[2])
      rect.size.height = CGFloat(boundingBox[3])
//        print(rect)

      // The detected corners are for model dimensions. So we scale the rect with respect to the
      // actual image dimensions.
      let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))

      // Gets the color assigned for the class
      let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
      let inference = Inference(confidence: score,
                                className: outputClass,
                                rect: newRect,
                                displayColor: colorToAssign)
      resultsArray.append(inference)
    }

    // Sort results in descending order of confidence.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultsArray
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
        labels = contents.components(separatedBy: "\n")
        labels.removeLast()
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
        
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
      free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)
    
    if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, vImage_Flags(Float32(kvImageNoFlags)))
    } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, vImage_Flags(Float32(kvImageNoFlags)))
    }

//    print(sourceBuffer)
//    let byteData = Data(bytes: sourceBuffer.data, count: sourceBuffer.rowBytes * height)
    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
//    print(byteData.count)
    if isModelQuantized {
      return byteData
    }
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
//    var pixel = 0
//    for i in 0..<inputWidth {
//        for j in 0..<inputHeight {
//            var myInt: UInt8 = 0xFF
//            let float1 = (((bytes[pixel]) >> 16) & myInt)
//            let float2 = (((bytes[pixel]) >> 8) & myInt)
//            let float3 = ((bytes[pixel]) & myInt)
//            floats.append(Float(float1) / imageStd)
//            floats.append(Float(float2) / imageStd)
//            floats.append(Float(float3) / imageStd)
//            pixel += 1
//        }
//    }
    for i in 0..<bytes.count {
      floats.append((Float(bytes[i]) - imageMean) / imageStd)
    }
    
//    let byteBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4 * batchSize * inputWidth * inputHeight * 3)
//    let intValues = Array<Int>(repeating: 0, count: inputHeight * inputWidth)
    


    // Not quantized, convert to floats
//    let bytes = Array<UInt8>(unsafeData: byteData)!
//    _ = Array<Float32>(unsafeData: byteData)!
//
//    let intValues = Array<Int>(repeating: 0, count: inputHeight * inputWidth)
//    for i in destinationBuffer.data.Z{
//        intValues[i] = j
//    }
//    var pixel = 0
//    var floats = [Float]()
//    for _ in 0..<inputWidth {
//        for _ in 0..<inputHeight {
//            let val = intValues[pixel]
//            pixel += 1
//            floats.append(Float((val >> 16) & 0xFF) / 255.0)
//            floats.append(Float((val >> 8) & 0xFF) / 255.0)
//            floats.append(Float(val & 0xFF) / 255.0)
//        }
//    }
    return Data(copyingBufferOf: floats)
  }
  func floatValue(data: Data) -> Float {
      return Float(bitPattern: UInt32(bigEndian: data.withUnsafeBytes { $0.load(as: UInt32.self) }))
  }

  /// This assigns color for a particular class.
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base
    // colors to each object based on its index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
