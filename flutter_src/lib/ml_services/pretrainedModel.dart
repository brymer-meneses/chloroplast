import 'classifier.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class PretrainedModel extends Classifier {
  PretrainedModel({int numThreads}) : super(numThreads: numThreads);
  TensorImage _inputImage;

  @override
  String get modelName => 'EfficientNet-Lite4.tflite';
  @override
  String get labelsFileName => '';
  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 255);
  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  // ignore: missing_return
  Future<void> loadLabels() {}

  TensorBuffer run(Image image) {
    _inputImage = TensorImage.fromImage(image);
    _inputImage = preProcess();

    interpreter.run(_inputImage.buffer, outputBuffer.getBuffer());

    return outputBuffer;
  }
}
