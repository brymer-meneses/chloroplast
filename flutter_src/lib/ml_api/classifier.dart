import 'dart:io';
import 'dart:math';

import 'package:Plant_Doctor/services/logger.dart';

import 'package:image/image.dart';
import 'package:collection/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

// For debugging purposes
Logger log = Logger(showLogs: true);

abstract class Classifier {
  Classifier({int numThreads}) {
    InterpreterOptions _interpreterOptions = InterpreterOptions();
    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
  }

  // ML Variables
  String get modelPath;
  String get labelsPath;
  Interpreter interpreter;
  List<int> inputShape, outputShape;

  TfLiteType outputType = TfLiteType.uint8;
  TensorBuffer outputBuffer;

  NormalizeOp get preProcessNormalizeOp;
  NormalizeOp get postProcessNormalizeOp;

  var probabilityProcessor;

  Future<void> loadModel() async {
    try {
      final String modelName = (modelPath.split('/')).length > 1
          ? modelPath.split('/').last
          : modelPath.split('/')[0];

      log("loadModel[1] Loading Model: $modelName");
      interpreter = await Interpreter.fromAsset(modelPath);
      log("loadModel[2] Successfully Loaded: $modelName");

      log("loadModel[3] Getting Information from $modelName");
      inputShape = interpreter.getInputTensor(0).shape;
      outputShape = interpreter.getOutputTensor(0).shape;
      outputType = interpreter.getOutputTensor(0).type;

      log("\nModel Name: $modelName");
      log("\t inputShape: $inputShape \n \t outputShape: $outputShape \n \t outputType: $outputType");

      log("\nloadModel[4] Instantiating essential variables");

      outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
      probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
      log("loadModel[5] Success!");
    } catch (e) {
      log("The function loadModel failed. Unable to create interpreter. Caught Exception: ${e.toString()}");
    }
  }

  List<String> labels;
  Future<void> loadLabels() async {
    try {
      log("loadLabels[1] Loading Labels from $labelsPath");
      labels = await FileUtil.loadLabels(labelsPath);
      log("loadLabels[2] Labels loaded successfully");
      log("loadLabels[3] Labels: $labels");
    } catch (e) {
      log("The function loadLabels failed. Caught Exception: ${e.toString}");
    }
  }

  static TensorImage preProcess(
    TensorImage inputImage,
    List<int> inputShape,
    NormalizeOp preProcessNormalizeOp,
  ) {
    log("preProcess[1] Performing preprocessing to the image");
    int cropSize = min(inputImage.height, inputImage.width);
    TensorImage preProccessedImage = ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            inputShape[1], inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(preProcessNormalizeOp)
        .build()
        .process(inputImage);
    log("preProcess[2] Successfully preprocessed image");
    return preProccessedImage;
  }

  TensorBuffer runOnImage(Image image) {
    if (interpreter == null) {
      throw StateError(
          'The function predict failed. Cannot run inference, Intrepreter is null');
    }
    log("runOnImage[1] Processing Image");
    TensorImage inputImage = preProcess(
        TensorImage.fromImage(image), inputShape, preProcessNormalizeOp);
    log("runOnImage[2] Running Inference");
    interpreter.run(inputImage.buffer, outputBuffer.getBuffer());
    return outputBuffer;
  }

  TensorBuffer runOnTensor(TensorBuffer input) {
    if (interpreter == null) {
      throw StateError(
          'The function predict failed. Cannot run inference, Intrepreter is null');
    }
    interpreter.run(input.buffer, outputBuffer.getBuffer());
    return outputBuffer;
  }

  Category getResults(TensorBuffer _outputBuffer) {
    Map<String, double> labeledProb = TensorLabel.fromList(
            labels, probabilityProcessor.process(_outputBuffer))
        .getMapWithFloatValue();
    final pred = getTopProbability(labeledProb);

    return Category(pred.key, pred.value);
  }
}

MapEntry<String, double> getTopProbability(Map<String, double> labeledProb) {
  var pq = PriorityQueue<MapEntry<String, double>>(compare);
  pq.addAll(labeledProb.entries);

  return pq.first;
}

int compare(MapEntry<String, double> e1, MapEntry<String, double> e2) {
  if (e1.value > e2.value) {
    return -1;
  } else if (e1.value == e2.value) {
    return 0;
  } else {
    return 1;
  }
}
