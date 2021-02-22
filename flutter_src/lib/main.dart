// Packages
import 'dart:io';
import 'package:Plant_Doctor/ml_services/pretrainedModel.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

// Other Files
import './services/imageBox.dart';
import './ml_services/pretrainedModel.dart';

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
  PretrainedModel _efficientNet = new PretrainedModel();

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

  void _startClassifyingImage() async {}

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
              ? Container(
                  child: FloatingActionButton.extended(
                  onPressed: () {},
                  backgroundColor: Colors.blueGrey,
                  label: Text('Classify Image'),
                ))
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
