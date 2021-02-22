import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart';

import 'classifier.dart';
import 'pretrainedModel.dart';

class PlantModel extends Classifier {
  final String _plantName;
  final PretrainedModel _pretrainedModel;

  PlantModel(this._plantName, this._pretrainedModel) {
    loadLabels();
  }

  @override
  String get labelsFileName => "plant_models/labels/$_plantName.txt";

  @override
  String get modelName => 'plant_models/models/$_plantName.tflite';

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  Category predict(Image image) {
    TensorBuffer _pretrainedModelOutput = _pretrainedModel.run(image);
    interpreter.run(_pretrainedModelOutput, outputBuffer.getBuffer());

    Map<String, double> labeledProb =
        TensorLabel.fromList(labels, probabilityProcessor.process(outputBuffer))
            .getMapWithFloatValue();
    final pred = getTopProbability(labeledProb);

    return Category(pred.key, pred.value);
  }
}
