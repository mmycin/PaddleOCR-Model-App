// lib/nets/detection.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'package:image/image.dart' as img;

/// PaddleOCR Text Detection — detects text regions in images
/// Simplified version that works with available APIs
class Detection {
  late final ort.OrtSession _session;
  late final String _inputName;
  final int minSize = 3;
  final int maxSize = 960;
  final double boxThresh = 0.8;
  final double maskThresh = 0.8;
  
  // ImageNet normalization
  final List<double> mean = [123.675, 116.28, 103.53];
  final List<double> std = [1 / 58.395, 1 / 57.12, 1 / 57.375];

  Detection.fromBuffer(Uint8List modelData) {
    ort.OrtEnv.instance.init();
    final options = ort.OrtSessionOptions();
    
    print('Detection: Loading model from buffer (${modelData.length} bytes)');
    
    try {
      _session = ort.OrtSession.fromBuffer(modelData, options);
      _inputName = _session.inputNames.first;
      print('Detection: Model loaded successfully');
    } catch (e) {
      throw Exception('Failed to load detection model from buffer: $e');
    }
  }

  Detection(String modelPath) {
    ort.OrtEnv.instance.init();
    final options = ort.OrtSessionOptions();
    
    // Verify file exists and get absolute path
    final file = File(modelPath);
    if (!file.existsSync()) {
      throw Exception('Detection model file does not exist: $modelPath');
    }
    
    // Use absolute path to avoid encoding issues
    final absolutePath = file.absolute.path;
    print('Detection: Loading model from absolute path: $absolutePath');
    
    try {
      _session = ort.OrtSession.fromFile(File(absolutePath), options);
      _inputName = _session.inputNames.first;
    } catch (e) {
      throw Exception('Failed to load detection model from $absolutePath: $e');
    }
  }

  /// Detects text regions in image, returns list of polygon points
  Future<List<List<List<double>>>> call(img.Image image) async {
    try {
      print('Starting text region detection...');
      final h = image.height;
      final w = image.width;

      // Zero pad if too small
      img.Image processed = image;
      if (h + w < 64) {
        processed = _zeroPad(image);
      }

      // Resize
      processed = _resize(processed);
      print('Image preprocessed: ${processed.width}x${processed.height}');

      // Preprocess
      final input = _preprocess(processed);
      
      // Run inference
      final runOpts = ort.OrtRunOptions();
      final outputs = _session.run(runOpts, {_inputName: input});
      print('Inference completed');
      
      // Extract and process detection output
      final outputValue = outputs[0];
      List<List<double>>? outputData;
      
      if (outputValue != null && outputValue is ort.OrtValueTensor) {
        final tensorValue = outputValue.value;
        if (tensorValue is List) {
          outputData = _convertTo2DMap(tensorValue);
          print('Output shape: ${outputData.length}x${outputData.isNotEmpty ? outputData[0].length : 0}');
        }
      }
      
      final boxes = outputData != null ? _extractBoxesFromData(outputData, image.width, image.height) : <List<List<double>>>[];
      print('Detection completed: ${boxes.length} text regions found');
      
      // Clean up resources
      input.release();
      runOpts.release();
      for (var output in outputs) {
        output?.release();
      }

      return boxes;
    } catch (e) {
      print('Detection error: $e');
      return <List<List<double>>>[]; // Return empty list on error
    }
  }

  /// Extracts text bounding boxes from model output
  List<List<List<double>>> _extractBoxes(dynamic output, int origWidth, int origHeight) {
    if (output == null) return [];
    
    try {
      // Get output data as List
      final outputData = output.value;
      if (outputData is! List) return [];
      
      // Convert to 2D probability map (assuming output shape is [1, 1, H, W])
      final probMap = _convertTo2DMap(outputData);
      if (probMap.isEmpty) return [];
      
      // Apply threshold to get binary map
      final binaryMap = _applyThreshold(probMap, maskThresh);
      
      // Find contours and extract bounding boxes
      final boxes = _findContours(binaryMap, origWidth, origHeight);
      
      return boxes;
    } catch (e) {
      print('Error extracting boxes: $e');
      return [];
    }
  }

  /// Extracts text bounding boxes from converted data
  List<List<List<double>>> _extractBoxesFromData(List<List<double>> probMap, int origWidth, int origHeight) {
    try {
      if (probMap.isEmpty) return [];
      
      // Apply threshold to get binary map
      final binaryMap = _applyThreshold(probMap, maskThresh);
      
      // Find contours and extract bounding boxes
      final boxes = _findContours(binaryMap, origWidth, origHeight);
      
      return boxes;
    } catch (e) {
      print('Error extracting boxes from data: $e');
      return [];
    }
  }

  /// Converts model output to 2D probability map
  List<List<double>> _convertTo2DMap(List outputData) {
    try {
      // Handle different output shapes
      if (outputData.isEmpty) return [];
      
      // If it's a flat list, try to reshape based on typical DBNet output
      if (outputData[0] is double) {
        final flatData = outputData.cast<double>();
        
        // Assume square output for now (common for text detection)
        final size = math.sqrt(flatData.length).round();
        if (size * size == flatData.length) {
          final map = List<List<double>>.generate(size, (i) {
            return flatData.sublist(i * size, (i + 1) * size);
          });
          return map;
        }
      }
      
      // If it's already 2D
      if (outputData[0] is List && outputData[0][0] is double) {
        return outputData.map((row) => (row as List).cast<double>()).toList();
      }
      
      return [];
    } catch (e) {
      print('Error converting to 2D map: $e');
      return [];
    }
  }

  /// Applies threshold to create binary map
  List<List<bool>> _applyThreshold(List<List<double>> probMap, double threshold) {
    return probMap.map((row) {
      return row.map((prob) => prob >= threshold).toList();
    }).toList();
  }

  /// Finds contours and extracts bounding boxes
  List<List<List<double>>> _findContours(List<List<bool>> binaryMap, int origWidth, int origHeight) {
    final boxes = <List<List<double>>>[];
    final visited = List.generate(binaryMap.length, (_) => List<bool>.filled(binaryMap[0].length, false));
    
    // Scale factors
    final scaleX = origWidth / binaryMap[0].length;
    final scaleY = origHeight / binaryMap.length;
    
    // Find connected components
    for (int y = 0; y < binaryMap.length; y++) {
      for (int x = 0; x < binaryMap[y].length; x++) {
        if (binaryMap[y][x] && !visited[y][x]) {
          final component = _floodFill(binaryMap, visited, x, y);
          if (component.length >= minSize) {
            final box = _componentToBox(component, scaleX, scaleY);
            if (box != null && _isValidBox(box)) {
              boxes.add(box);
            }
          }
        }
      }
    }
    
    return boxes;
  }

  /// Flood fill algorithm to find connected components
  List<List<int>> _floodFill(List<List<bool>> binaryMap, List<List<bool>> visited, int startX, int startY) {
    final component = <List<int>>[];
    final stack = <List<int>>[[startX, startY]];
    final directions = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]];
    
    while (stack.isNotEmpty) {
      final point = stack.removeLast();
      final x = point[0];
      final y = point[1];
      
      if (x < 0 || x >= binaryMap[0].length || y < 0 || y >= binaryMap.length) continue;
      if (!binaryMap[y][x] || visited[y][x]) continue;
      
      visited[y][x] = true;
      component.add([x, y]);
      
      // Add neighbors
      for (final dir in directions) {
        stack.add([x + dir[0], y + dir[1]]);
      }
    }
    
    return component;
  }

  /// Converts component to bounding box quadrilateral
  List<List<double>>? _componentToBox(List<List<int>> component, double scaleX, double scaleY) {
    if (component.isEmpty) return null;
    
    // Find min/max coordinates
    int minX = component[0][0], maxX = component[0][0];
    int minY = component[0][1], maxY = component[0][1];
    
    for (final point in component) {
      if (point[0] < minX) minX = point[0];
      if (point[0] > maxX) maxX = point[0];
      if (point[1] < minY) minY = point[1];
      if (point[1] > maxY) maxY = point[1];
    }
    
    // Scale to original image coordinates
    final scaledMinX = minX * scaleX;
    final scaledMaxX = maxX * scaleX;
    final scaledMinY = minY * scaleY;
    final scaledMaxY = maxY * scaleY;
    
    // Create quadrilateral (4-point polygon)
    return [
      [scaledMinX, scaledMinY], // top-left
      [scaledMaxX, scaledMinY], // top-right
      [scaledMaxX, scaledMaxY], // bottom-right
      [scaledMinX, scaledMaxY], // bottom-left
    ];
  }

  /// Validates if a box is reasonable
  bool _isValidBox(List<List<double>> box) {
    if (box.length != 4) return false;
    
    // Calculate dimensions
    final width = (box[1][0] - box[0][0]).abs();
    final height = (box[3][1] - box[0][1]).abs();
    
    // Filter out too small or too large boxes
    if (width < 5 || height < 5) return false; // Too small
    if (width > 2000 || height > 500) return false; // Too large
    
    // Filter out extreme aspect ratios
    final aspectRatio = width / height;
    if (aspectRatio > 50 || aspectRatio < 0.02) return false;
    
    return true;
  }

  img.Image _resize(img.Image image) {
    final h = image.height;
    final w = image.width;

    double ratio = 1.0;
    if (math.max(h, w) > maxSize) {
      if (h > w) {
        ratio = maxSize / h;
      } else {
        ratio = maxSize / w;
      }
    }

    final resizeH = math.max((h * ratio / 32).round() * 32, 32);
    final resizeW = math.max((w * ratio / 32).round() * 32, 32);

    return img.copyResize(image, width: resizeW, height: resizeH);
  }

  img.Image _zeroPad(img.Image image) {
    final h = image.height;
    final w = image.width;
    final c = image.numChannels;
    
    final padH = math.max(32, h);
    final padW = math.max(32, w);
    
    final padded = img.Image(width: padW, height: padH, numChannels: c);
    // Copy image into padded canvas
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        padded.setPixel(x, y, image.getPixel(x, y));
      }
    }
    
    return padded;
  }

  ort.OrtValueTensor _preprocess(img.Image image) {
    final h = image.height;
    final w = image.width;
    
    // Convert to CHW format and normalize
    final buffer = Float32List(3 * h * w);
    
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final pixel = image.getPixel(x, y);
          double value = 0.0;
          if (c == 0) value = pixel.r.toDouble();
          else if (c == 1) value = pixel.g.toDouble();
          else value = pixel.b.toDouble();
          
          // Normalize: (value - mean) * std
          value = (value - mean[c]) * std[c];
          buffer[c * h * w + y * w + x] = value;
        }
      }
    }

    // Create tensor from Float32List
    // IMPORTANT: Pass Float32List directly to createTensorWithDataList
    // The API should accept Float32List and create float32 tensor (not double)
    // According to onnxruntime package docs, createTensorWithDataList accepts Float32List
    return ort.OrtValueTensor.createTensorWithDataList(
      buffer,
      [1, 3, h, w],
    );
  }

  void dispose() {
    _session.release();
  }
}