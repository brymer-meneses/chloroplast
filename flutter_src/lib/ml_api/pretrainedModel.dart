import 'classifier.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart';

class PretrainedModel extends Classifier {
  PretrainedModel({int numThreads}) : super(numThreads: numThreads) {
    loadModel();
  }

  TensorImage _inputImage;

  @override
  String get modelPath => 'EfficientNet-Lite4.tflite';

  @override
  String get labelsPath => '';

  @override
  // Mimic the image processing that was used on the training data
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 255);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  TensorBuffer run(Image image) {
    _inputImage = Classifier.preProcess(
        TensorImage.fromImage(image), inputShape, preProcessNormalizeOp);

    interpreter.run(_inputImage.buffer, outputBuffer.getBuffer());

    return outputBuffer;
  }
}
