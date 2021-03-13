import 'dart:collection';

import './plantModel.dart';
import './pretrainedModel.dart';
import '../services/logger.dart';

Logger log = Logger(showLogs: true);

class InitializeModels {
  HashMap plantModels = new HashMap<String, PlantModel>();

  final PretrainedModel _efficientNet;
  static List<String> models = [
    'apple',
    'corn',
    'cherry',
    'grape',
    'pepper',
    'potato',
    'strawberry'
  ];

  InitializeModels(this._efficientNet) {
    for (var i = 0; i < models.length; i++) {
      try {
        plantModels[models[i]] = new PlantModel(models[i], _efficientNet);
      } catch (e) {
        log("[Error] Unable to load ${models[i]} caught exception: ${e.toString()}")
      }
    }
  }

  HashMap<String, PlantModel> getModels() {
    return this.plantModels;
  }
}
