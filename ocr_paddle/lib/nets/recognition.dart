// lib/nets/recognition.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnxruntime/onnxruntime.dart' as ort;
import 'package:image/image.dart' as img;
import '../utils/utils.dart' show CTCDecoder;

/// PaddleOCR Text Recognition — recognizes text from cropped images
class Recognition {
  late final ort.OrtSession _session;
  late final String _inputName;
  final List<int> inputShape = [3, 48, 320];
  final CTCDecoder ctcDecoder = CTCDecoder();

  Recognition.fromBuffer(Uint8List modelData) {
    ort.OrtEnv.instance.init();
    final options = ort.OrtSessionOptions();
    // options.addCUDA(); // Uncomment if built with CUDA
    
    print('Recognition: Loading model from buffer (${modelData.length} bytes)');

    try {
      _session = ort.OrtSession.fromBuffer(modelData, options);
      _inputName = _session.inputNames.first;
      print('Recognition: Model loaded successfully');
    } catch (e) {
      throw Exception('Failed to load recognition model from buffer: $e');
    }
  }

  Recognition(String modelPath) {
    ort.OrtEnv.instance.init();
    final options = ort.OrtSessionOptions();
    // options.addCUDA(); // Uncomment if built with CUDA
    
    // Verify file exists and get absolute path
    final file = File(modelPath);
    if (!file.existsSync()) {
      throw Exception('Recognition model file does not exist: $modelPath');
    }
    
    // Use absolute path to avoid encoding issues
    final absolutePath = file.absolute.path;
    print('Recognition: Loading model from absolute path: $absolutePath');

    try {
      _session = ort.OrtSession.fromFile(File(absolutePath), options);
      _inputName = _session.inputNames.first;
    } catch (e) {
      throw Exception('Failed to load recognition model from $absolutePath: $e');
    }
  }

  /// Recognizes text from cropped images
  /// Returns: Map with 'results' (List<String>) and 'confidences' (List<List<double>>)
  Future<Map<String, List<dynamic>>> call(List<img.Image> images) async {
    final batchSize = 6;
    final numImages = images.length;
    
    final results = List<String>.filled(numImages, '');
    final confidences = List<List<double>>.filled(numImages, []);

    // Sort by aspect ratio
    final indices = List<int>.generate(numImages, (i) => i)
      ..sort((a, b) {
        final ratioA = images[a].width / images[a].height;
        final ratioB = images[b].width / images[b].height;
        return ratioA.compareTo(ratioB);
      });

    for (var index = 0; index < numImages; index += batchSize) {
      final inputH = inputShape[1];
      final inputW = inputShape[2];
      double maxWhRatio = inputW / inputH;

      // Find max width/height ratio in batch
      for (var i = index; i < math.min(numImages, index + batchSize); i++) {
        final img = images[indices[i]];
        final ratio = img.width / img.height;
        if (ratio > maxWhRatio) maxWhRatio = ratio;
      }

      // Preprocess images
      final normImages = <Float32List>[];
      for (var i = index; i < math.min(numImages, index + batchSize); i++) {
        final normImage = _resize(images[indices[i]], maxWhRatio);
        normImages.add(normImage);
      }

      // Create batch tensor
      final input = _makeTensor(normImages);
      
      // Run inference
      final runOpts = ort.OrtRunOptions();
      final outputs = _session.run(runOpts, {_inputName: input});
      
      // Decode CTC outputs
      final decoded = ctcDecoder.call(outputs[0]);
      final decodedResults = decoded['results'] as List<String>;
      final decodedConfidences = decoded['confidences'] as List<List<double>>;

      // Store results in original order
      for (var i = 0; i < decodedResults.length; i++) {
        final origIdx = indices[index + i];
        results[origIdx] = decodedResults[i];
        confidences[origIdx] = decodedConfidences[i];
      }

      input.release();
      runOpts.release();
      for (var output in outputs) {
        output?.release();
      }
    }

    return {'results': results, 'confidences': confidences};
  }

  /// Resizes image and normalizes for recognition model
  Float32List _resize(img.Image image, double maxWhRatio) {
    final inputH = inputShape[1];
    final inputW = inputShape[2];
    final inputC = inputShape[0];

    final h = image.height;
    final w = image.width;
    final ratio = w / h;

    int resizedW;
    final ceilRatio = (inputH * ratio).ceil();
    if (ceilRatio > inputW) {
      resizedW = inputW;
    } else {
      resizedW = ceilRatio;
    }

    // Resize image
    final resized = img.copyResize(image, width: resizedW, height: inputH);

    // Convert to CHW format and normalize
    final buffer = Float32List(inputC * inputH * inputW);
    
    // Initialize padded buffer
    for (int c = 0; c < inputC; c++) {
      for (int y = 0; y < inputH; y++) {
        for (int x = 0; x < inputW; x++) {
          if (x < resizedW) {
            final pixel = resized.getPixel(x, y);
            double value = 0.0;
            if (c == 0) value = pixel.r.toDouble();
            else if (c == 1) value = pixel.g.toDouble();
            else value = pixel.b.toDouble();
            
            // Normalize: ((value / 255) - 0.5) / 0.5
            value = ((value / 255.0) - 0.5) / 0.5;
            buffer[c * inputH * inputW + y * inputW + x] = value;
          } else {
            buffer[c * inputH * inputW + y * inputW + x] = 0.0;
          }
        }
      }
    }

    return buffer;
  }

  /// Creates tensor from list of preprocessed images
  ort.OrtValueTensor _makeTensor(List<Float32List> tensors) {
    final b = tensors.length;
    final inputH = inputShape[1];
    final inputW = inputShape[2];
    final inputC = inputShape[0];
    
    final buffer = Float32List(b * inputC * inputH * inputW);
    final elementSize = inputC * inputH * inputW;

    for (var i = 0; i < b; i++) {
      buffer.setRange(i * elementSize, (i + 1) * elementSize, tensors[i]);
    }

    // Create tensor from Float32List
    // IMPORTANT: Pass Float32List directly to createTensorWithDataList
    // The API should accept Float32List and create float32 tensor (not double)
    // According to onnxruntime package docs, createTensorWithDataList accepts Float32List
    return ort.OrtValueTensor.createTensorWithDataList(
      buffer,
      [b, inputC, inputH, inputW],
    );
  }

  void dispose() {
    _session.release();
  }
}
