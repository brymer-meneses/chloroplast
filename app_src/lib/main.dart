// Packages
import 'dart:collection';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;

// Other Files
import './widgets/imageBox.dart';
import './ml_api/plantModel.dart';
import './ml_api/pretrainedModel.dart';
import './ml_api/initializeModels.dart';

void main() => runApp(MainApp());

class MainApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.light(),
      home: HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File _image;
  final PretrainedModel _efficientNet = new PretrainedModel();

  bool areModelsReady = false;
  HashMap<String, PlantModel> models;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

    var initializer = InitializeModels(_efficientNet);
    models = initializer.getModels();

    areModelsReady = true;
  }

  void _getImageFromGallery() async {
    final picker = ImagePicker();
    var pickedImage = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      _image = File(pickedImage.path);
    });
  }

  void _getImageFromCamera() async {
    final picker = ImagePicker();
    var pickedImage = await picker.getImage(source: ImageSource.camera);

    setState(() {
      _image = File(pickedImage.path);
    });
  }

  void _startClassifyingImage(File image, String modelName) async {
    // TODO 
    // Create a button to select what model will be used to 
    // classifiy the imageo

    PlantModel plantModel = models[modelName]
    var predictions = plantModel.runInference(image);
    print(predictions);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Plant Doctor"),
        centerTitle: true,
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          ImageBox(_image),
          _image != null
              ? Center(
                  child: Container(
                    child: FloatingActionButton.extended(
                      onPressed: () => _startClassifyingImage(_image, 'apple'), // Temporary
                      backgroundColor: Colors.blueGrey,
                      label: Text('Classify Image'),
                    ),
                  ),
                )
              : Container()
        ],
      ),
      floatingActionButton: Container(
          margin: EdgeInsets.all(5),
          child: FloatingActionButton.extended(
            label: Text('Choose an image'),
            isExtended: true,
            backgroundColor: Colors.blue,
            onPressed: _getImageFromGallery,
            icon: Icon(Icons.add_a_photo),
          )),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }
}
