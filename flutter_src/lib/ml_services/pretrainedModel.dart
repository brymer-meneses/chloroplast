import 'dart:math';
import 'dart:io';

import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

abstract class PretrainedModel {
  Interpreter interpreter;
  final String _path;

  List<int> _inputShape;
  List<int> _outputShape;

  TensorImage _inputImage;
  TensorBuffer _outputBuffer;

  TfLiteType _outputType = TfLiteType.uint8;

  PretrainedModel(this._path);

  NormalizeOp get preProcessNormalizeOp;
  NormalizeOp get postProcessNormalizeOp;

  var _probabilityProcessor;

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(_path);
      print("Successfully loaded the interpreter");

      _inputShape = interpreter.getInputTensor(0).shape;
      _outputShape = interpreter.getOutputTensor(0).shape;
      _outputType = interpreter.getOutputTensor(0).type;

      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
      _probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
    } catch (e) {
      print("Failed to load the model, caught exception: ${e.toString()}");
    }
  }

  TensorImage _preProcessImage() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(preProcessNormalizeOp)
        .build()
        .process(_inputImage);
  }

  TensorBuffer run(Image image) {
    if (interpreter == null) {
      throw StateError('Cannot run inference, Interpreter is null');
    }
    _inputImage = TensorImage.fromImage(image);
    _inputImage = _preProcessImage();

    interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer);
    interpreter.close();
    return _outputBuffer;
  }
}
