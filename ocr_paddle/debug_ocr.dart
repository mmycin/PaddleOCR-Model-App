import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'lib/nets/detection.dart';
import 'lib/nets/classification.dart';
import 'lib/nets/recognition.dart';

void main() async {
  print('Starting OCR Debug Test...');
  
  try {
    // Test model loading
    print('Loading models...');
    final detectionModel = await rootBundle.load('assets/models/detection.onnx');
    final classificationModel = await rootBundle.load('assets/models/classification.onnx');
    final recognitionModel = await rootBundle.load('assets/models/recognition.onnx');
    
    print('Models loaded successfully:');
    print('- Detection model: ${detectionModel.lengthInBytes} bytes');
    print('- Classification model: ${classificationModel.lengthInBytes} bytes');
    print('- Recognition model: ${recognitionModel.lengthInBytes} bytes');
    
    // Initialize models
    print('Initializing models...');
    final detection = Detection.fromBuffer(detectionModel.buffer.asUint8List());
    final classification = Classification.fromBuffer(classificationModel.buffer.asUint8List());
    final recognition = Recognition.fromBuffer(recognitionModel.buffer.asUint8List());
    
    print('Models initialized successfully');
    
    // Test with a simple image
    print('Creating test image...');
    final testImage = img.Image(width: 200, height: 100);
    
    // Fill with white background
    for (int y = 0; y < testImage.height; y++) {
      for (int x = 0; x < testImage.width; x++) {
        testImage.setPixel(x, y, img.ColorRgb8(255, 255, 255));
      }
    }
    
    // Add some black pixels to simulate text
    for (int x = 10; x < 50; x++) {
      for (int y = 20; y < 30; y++) {
        testImage.setPixel(x, y, img.ColorRgb8(0, 0, 0));
      }
    }
    
    print('Test image created: ${testImage.width}x${testImage.height}');
    
    // Test detection
    print('Running detection...');
    final detections = await detection.call(testImage);
    print('Detection completed. Found ${detections.length} text regions');
    
    if (detections.isNotEmpty) {
      print('First detection: ${detections[0]}');
      
      // Test classification
      print('Running classification...');
      final croppedImages = [testImage]; // Simplified for testing
      final classified = await classification.call(croppedImages);
      print('Classification completed');
      
      // Test recognition
      print('Running recognition...');
      final recognitionResults = await recognition.call(classified);
      print('Recognition completed: $recognitionResults');
    } else {
      print('No text regions detected in test image');
    }
    
    print('Debug test completed successfully');
    
  } catch (e, stackTrace) {
    print('Error during debug test: $e');
    print('Stack trace: $stackTrace');
  }
}