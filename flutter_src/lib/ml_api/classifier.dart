// Modified From:
// https://github.com/am15h/tflite_flutter_helper/blob/master/example/image_classification/lib/classifier.dart
// Copyright 2020, Amish Garg, All Rights Reserved
// Licensed under the Apache License, Version 2.0 (the "License")

import 'dart:io';
import 'dart:math';

import 'package:image/image.dart';
import 'package:collection/collection.dart';
import 'package:logger/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

abstract class Classifier {
  Interpreter interpreter;
  InterpreterOptions _interpreterOptions;

  var logger = Logger();

  List<int> inputShape;
  List<int> outputShape;

  TensorBuffer outputBuffer;

  TfLiteType outputType = TfLiteType.uint8;

  final int _labelsLength = 1001;

  var probabilityProcessor;

  List<String> labels;

  String get modelName;
  String get labelsFileName;

  NormalizeOp get preProcessNormalizeOp;
  NormalizeOp get postProcessNormalizeOp;

  Classifier({int numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }

    loadModel();
    // loadLabels();
  }

  Future<void> loadModel() async {
    try {
      print('Loading Model: $modelName');
      interpreter =
          await Interpreter.fromAsset(modelName, options: _interpreterOptions);
      print('The interpreter for $modelName was created successfully');

      inputShape = interpreter.getInputTensor(0).shape;
      outputShape = interpreter.getOutputTensor(0).shape;
      outputType = interpreter.getOutputTensor(0).type;

      outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
      probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  Future<void> loadLabels() async {
    labels = await FileUtil.loadLabels(labelsFileName);
    if (labels.length == _labelsLength) {
      print('Labels loaded successfully');
    } else {
      print(
          'Unable to load labels expected: $_labelsLength got ${labels.length}');
    }
  }

  TensorImage preProcess(TensorImage inputImage) {
    int cropSize = min(inputImage.height, inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            inputShape[1], inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(preProcessNormalizeOp)
        .build()
        .process(inputImage);
  }

  Category predict(Image image) {
    if (interpreter == null) {
      throw StateError('Cannot run inference, Intrepreter is null');
    }
    final pres = DateTime.now().millisecondsSinceEpoch;
    TensorImage inputImage = preProcess(TensorImage.fromImage(image));

    final pre = DateTime.now().millisecondsSinceEpoch - pres;

    print('Time to load image: $pre ms');

    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(inputImage.buffer, outputBuffer.getBuffer());
    final run = DateTime.now().millisecondsSinceEpoch - runs;

    print('Time to run inference: $run ms');

    Map<String, double> labeledProb =
        TensorLabel.fromList(labels, probabilityProcessor.process(outputBuffer))
            .getMapWithFloatValue();
    final pred = getTopProbability(labeledProb);

    return Category(pred.key, pred.value);
  }

  void close() {
    if (interpreter != null) {
      interpreter.close();
    }
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
