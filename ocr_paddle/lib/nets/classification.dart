// lib/nets/classification.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'package:image/image.dart' as img;

/// PaddleOCR Angle Classifier — rotates upside-down text
class Classification {
  late final ort.OrtSession _session;
  late final String _inputName;
  final double threshold = 0.98;

  Classification.fromBuffer(Uint8List modelData) {
    // Must init OrtEnv first
    ort.OrtEnv.instance.init();

    final options = ort.OrtSessionOptions();
    // options.addCUDA(); // Uncomment if built with CUDA

    print('Classification: Loading model from buffer (${modelData.length} bytes)');

    try {
      _session = ort.OrtSession.fromBuffer(modelData, options);
      _inputName = _session.inputNames.first;
      print('Classification: Model loaded successfully');
    } catch (e) {
      throw Exception('Failed to load classification model from buffer: $e');
    }
  }

  Classification(String modelPath) {
    // Must init OrtEnv first
    ort.OrtEnv.instance.init();

    final options = ort.OrtSessionOptions();
    // options.addCUDA(); // Uncomment if built with CUDA

    // Verify file exists and get absolute path
    final file = File(modelPath);
    if (!file.existsSync()) {
      throw Exception('Classification model file does not exist: $modelPath');
    }

    // Use absolute path to avoid encoding issues
    final absolutePath = file.absolute.path;
    print('Classification: Loading model from absolute path: $absolutePath');

    try {
      _session = ort.OrtSession.fromFile(File(absolutePath), options);
      _inputName = _session.inputNames.first;
    } catch (e) {
      throw Exception('Failed to load classification model from $absolutePath: $e');
    }
  }

  /// Input: List of cropped text images
  /// Output: Same list, rotated 180° if needed
  Future<List<img.Image>> call(List<img.Image> crops) async {
    if (crops.isEmpty) return crops;

    final order = List<int>.generate(crops.length, (i) => i)
      ..sort(
        (a, b) => (crops[a].width / crops[a].height).compareTo(
          crops[b].width / crops[b].height,
        ),
      );

    final result = List<img.Image>.from(crops);

    const batchSize = 6;
    for (var i = 0; i < order.length; i += batchSize) {
      final batch = order.sublist(i, math.min(i + batchSize, order.length));
      final tensors = batch.map((idx) => _preprocess(crops[idx])).toList();

      final input = _makeTensor(tensors);
      final runOpts = ort.OrtRunOptions();
      final outputs = _session.run(runOpts, {_inputName: input});

      // Parse outputs: shape is [batch, 2] where 2 is [p0, p180]
      final outputValue = outputs[0];
      List<List<double>> outputList;
      
      // Handle nested list structure
      if (outputValue is List) {
        final list = outputValue as List;
        if (list.isNotEmpty && list[0] is List) {
          outputList = list.map((e) => (e as List).cast<double>()).toList();
        } else {
          // Flat list, reshape to [batch, 2]
          final flat = list.cast<double>();
          outputList = [];
          for (int i = 0; i < flat.length; i += 2) {
            if (i + 1 < flat.length) {
              outputList.add([flat[i], flat[i + 1]]);
            }
          }
        }
      } else {
        outputList = [];
      }
      
      // Parse outputs
      for (var j = 0; j < batch.length && j < outputList.length; j++) {
        final idx = batch[j];
        final p0 = outputList[j][0];
        final p180 = outputList[j][1];
        
        if (p180 > p0 && p180 > threshold) {
          result[idx] = img.copyRotate(crops[idx], angle: 180);
        }
      }

      input.release();
      runOpts.release();
      for (var output in outputs) {
        output?.release();
      }
    }

    return result;
  }

  /// Image → CHW Float32 array [3,48,192]
  Float32List _preprocess(img.Image image) {
    final ratio = image.width / image.height;
    final w = (48 * ratio).ceil().clamp(1, 192);
    final resized = img.copyResize(image, width: w, height: 48);

    // Convert to CHW format and normalize
    final buffer = Float32List(3 * 48 * 192);
    
    // Initialize padded buffer
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 48; h++) {
        for (int w_idx = 0; w_idx < 192; w_idx++) {
          if (w_idx < w) {
            final pixel = resized.getPixel(w_idx, h);
            double value = 0.0;
            if (c == 0) value = pixel.r.toDouble();
            else if (c == 1) value = pixel.g.toDouble();
            else value = pixel.b.toDouble();
            
            // Normalize: /255 - 0.5) / 0.5
            value = ((value / 255.0) - 0.5) / 0.5;
            buffer[c * 48 * 192 + h * 192 + w_idx] = value;
          } else {
            buffer[c * 48 * 192 + h * 192 + w_idx] = 0.0;
          }
        }
      }
    }

    return buffer;
  }

  /// List<Float32List> → OrtValueTensor [B,3,48,192]
  ort.OrtValueTensor _makeTensor(List<Float32List> tensors) {
    final b = tensors.length;
    final buffer = Float32List(b * 3 * 48 * 192);
    const elementSize = 3 * 48 * 192;

    for (var i = 0; i < b; i++) {
      buffer.setRange(i * elementSize, (i + 1) * elementSize, tensors[i]);
    }

    // Create tensor from Float32List
    // IMPORTANT: Pass Float32List directly to createTensorWithDataList
    // The API should accept Float32List and create float32 tensor (not double)
    // According to onnxruntime package docs, createTensorWithDataList accepts Float32List
    return ort.OrtValueTensor.createTensorWithDataList(
      buffer,
      [b, 3, 48, 192],
    );
  }

  void dispose() {
    _session.release();
  }
}