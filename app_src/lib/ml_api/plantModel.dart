import 'dart:io';

import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

import 'classifier.dart';
import 'pretrainedModel.dart';

class PlantModel extends Classifier {
  final String _plantName;
  final PretrainedModel _pretrainedModel;

  PlantModel(this._plantName, this._pretrainedModel, {int numThreads})
      : super(numThreads: numThreads) {
    loadModel();
    loadLabels();
  }

  @override
  String get labelsPath => "assets/plant_models/labels/$_plantName.txt";
  // flutter_src\assets\plant_models\models\apple.tflite
  @override
  String get modelPath => 'plant_models/models/$_plantName.tflite';

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  Map<String, double> runInference(File image) {
    img.Image imageInput = img.decodeImage(image.readAsBytesSync());
    TensorBuffer _pretrainedModelOutput =
        _pretrainedModel.runOnImage(imageInput);
    TensorBuffer _plantModelOutput = super.runOnTensor(_pretrainedModelOutput);

    Map<String, double> results = super.getResults(_plantModelOutput);
    return results;
  }
}
