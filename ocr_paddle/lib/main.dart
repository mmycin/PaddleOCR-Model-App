import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io' show File;
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';


import 'nets/detection.dart';
import 'nets/classification.dart';
import 'nets/recognition.dart';
import 'utils/utils.dart' show sortPolygon, cropImage;
import 'utils/image_splitter.dart';
import 'utils/performance_optimizer.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PaddleOCR Flutter',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const OCRScreen(),
    );
  }
}

class OCRScreen extends StatefulWidget {
  const OCRScreen({super.key});

  @override
  State<OCRScreen> createState() => _OCRScreenState();
}

class _OCRScreenState extends State<OCRScreen> {
  final ImagePicker _picker = ImagePicker();
  XFile? _pickedImage;
  List<String> _results = [];
  bool _isProcessing = false;
  String? _errorMessage;

  Detection? _detection;
  Classification? _classification;
  Recognition? _recognition;
  late PerformanceOptimizer _optimizer;

  @override
  void initState() {
    super.initState();
    _optimizer = PerformanceOptimizer();
    _loadModels();
  }
  
  @override
  void dispose() {
    _optimizer.dispose();
    super.dispose();
  }

  Future<void> _loadModels() async {
    try {
      setState(() {
        _isProcessing = true;
        _errorMessage = null;
      });

      print('Loading models from assets...');
      // Load models from assets
      final detectionModel = await _loadAsset('assets/models/detection.onnx');
      final classificationModel = await _loadAsset('assets/models/classification.onnx');
      final recognitionModel = await _loadAsset('assets/models/recognition.onnx');

      print('Models loaded from assets successfully');
      print('Detection model size: ${detectionModel.length} bytes');
      print('Classification model size: ${classificationModel.length} bytes');
      print('Recognition model size: ${recognitionModel.length} bytes');

      // Load models directly from memory to avoid file path encoding issues
      print('Initializing detection model...');
      _detection = Detection.fromBuffer(detectionModel);
      print('Detection model initialized');
      
      print('Initializing classification model...');
      _classification = Classification.fromBuffer(classificationModel);
      print('Classification model initialized');
      
      print('Initializing recognition model...');
      _recognition = Recognition.fromBuffer(recognitionModel);
      print('Recognition model initialized');

      setState(() {
        _isProcessing = false;
      });
      print('All models loaded successfully');
    } catch (e, stackTrace) {
      print('Error loading models: $e');
      print('Stack trace: $stackTrace');
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error loading models: $e\n\nDetails: $stackTrace';
      });
    }
  }

  Future<Uint8List> _loadAsset(String path) async {
    final byteData = await rootBundle.load(path);
    return byteData.buffer.asUint8List();
  }

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          _pickedImage = image;
          _results = [];
          _errorMessage = null;
        });
        await _processImage();
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error picking image: $e';
      });
    }
  }

  Future<void> _captureImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.camera);
      if (image != null) {
        setState(() {
          _pickedImage = image;
          _results = [];
          _errorMessage = null;
        });
        await _processImage();
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error capturing image: $e';
      });
    }
  }

  Future<void> _processImage() async {
    if (_pickedImage == null || _detection == null || _classification == null || _recognition == null) {
      setState(() {
        _errorMessage = 'Models not loaded properly. Please wait for models to load.';
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      print('Starting image processing...');
      // Load image
      final imageBytes = await File(_pickedImage!.path).readAsBytes();
      print('Image loaded: ${imageBytes.length} bytes');
      
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image - invalid image format');
      }
      
      print('Image decoded: ${image.width}x${image.height}');

      // Validate image quality
      final splitter = ImageSplitter();
      print('Validating image quality...');
      if (!splitter.validateImage(image)) {
        print('Image validation failed - image too dark or low contrast');
        throw Exception('Image quality too low for text detection. Please use a clearer image.');
      }
      print('Image validation passed');

      // Image package already uses RGB, so use as is
      final rgbImage = image;

      // Step 1: Optimize image for OCR processing
      final optimizedImage = _optimizer.optimizeImageForOCR(rgbImage);
      
      // Check cache for preprocessed image
      final cacheKey = 'ocr_${optimizedImage.width}_${optimizedImage.height}_${DateTime.now().millisecondsSinceEpoch}';
      final cachedImage = _optimizer.getCachedImage(cacheKey);
      final processedImage = cachedImage ?? optimizedImage;
      
      if (cachedImage == null) {
        _optimizer.cacheImage(cacheKey, optimizedImage);
      }
      
      // Split large images into tiles for better detection
      final tiles = splitter.splitImage(processedImage);
      final tilePositions = <Map<String, int>>[];
      final tileDetections = <List<List<List<double>>>>[];

      // Process tiles in batches for better performance
      print('Processing ${tiles.length} tiles with detection model...');
      final batchResults = await _optimizer.batchProcess(
        tiles,
        (tile) async {
          print('Processing tile: ${tile.width}x${tile.height}');
          final tilePoints = await _detection!.call(tile);
          print('Detection found ${tilePoints.length} text regions in tile');
          return tilePoints;
        },
      );
      print('Batch processing completed. Total results: ${batchResults.length}');
      
      // Process batch results and adjust coordinates
      for (int i = 0; i < batchResults.length; i++) {
        final tilePoints = batchResults[i];
        final tilePosition = splitter.calculateTilePosition(i, processedImage.width, processedImage.height);
        
        tilePositions.add(tilePosition);
        tileDetections.add(tilePoints);
      }

      // Merge detections from all tiles
      final mergedPoints = splitter.mergeDetections(
        tileDetections,
        tilePositions,
        rgbImage.width,
        rgbImage.height,
      );

      // Fallback to single image processing if no tiles were created
      print('Merged points: ${mergedPoints.length}');
      final points = mergedPoints.isEmpty && tiles.length == 1 
          ? await _detection!.call(rgbImage)
          : mergedPoints;
      
      print('Total detection points: ${points.length}');
      
      if (points.isEmpty) {
        print('No text regions detected');
        setState(() {
          _results = ['No text detected in image. Please try a clearer image with visible text.'];
          _isProcessing = false;
        });
        return;
      }
      
      print('Detected ${points.length} text regions');

      // Step 2: Sort polygons - sort points within each polygon
      final sortedPoints = points.map((polygon) {
        // Validate polygon
        if (polygon.isEmpty || polygon.length < 4) {
          return polygon;
        }
        
        // Convert polygon from List<List<double>> to List<List<double>> for sortPolygon
        final polygonPoints = polygon.map((pt) => [pt[0], pt[1]]).toList();
        return sortPolygon(polygonPoints);
      }).where((polygon) => polygon.isNotEmpty && polygon.length >= 4).toList();

      if (sortedPoints.isEmpty) {
        setState(() {
          _results = ['No valid text regions found.'];
          _isProcessing = false;
        });
        return;
      }

      // Step 3: Crop images with error handling
      final croppedImages = <img.Image>[];
      for (int i = 0; i < sortedPoints.length; i++) {
        try {
          final cropped = cropImage(rgbImage, sortedPoints[i]);
          if (cropped.width > 0 && cropped.height > 0) {
            croppedImages.add(cropped);
          }
        } catch (e) {
          // Skip this region if cropping fails
          continue;
        }
      }

      if (croppedImages.isEmpty) {
        setState(() {
          _results = ['Could not extract valid text regions from image.'];
          _isProcessing = false;
        });
        return;
      }

      // Step 4: Classification (rotate if needed)
      final rotatedImages = await _classification!.call(croppedImages);

      // Step 5: Recognition with confidence scores
      final recognitionResults = await _recognition!.call(rotatedImages);
      final results = recognitionResults['results'] as List<String>;
      final confidences = recognitionResults['confidences'] as List<List<double>>;

      // Filter results by confidence and remove empty strings
      final filteredResults = <String>[];
      for (int i = 0; i < results.length; i++) {
        final text = results[i].trim();
        final avgConfidence = confidences[i].isNotEmpty 
            ? confidences[i].reduce((a, b) => a + b) / confidences[i].length 
            : 0.0;
        
        if (text.isNotEmpty && avgConfidence > 0.5) {
          filteredResults.add(text);
        }
      }

      if (filteredResults.isEmpty) {
        setState(() {
          _results = ['Text detected but could not be recognized clearly. Please try a higher quality image.'];
          _isProcessing = false;
        });
        return;
      }

      setState(() {
        _results = filteredResults;
        _isProcessing = false;
      });
    } catch (e, stackTrace) {
      setState(() {
        _errorMessage = 'Error processing image: $e\n\nPlease try:\n• Using a clearer image\n• Ensuring text is visible\n• Checking image format (JPG/PNG recommended)';
        _isProcessing = false;
      });
    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('PaddleOCR Flutter'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image display
            if (_pickedImage != null)
              Container(
                margin: const EdgeInsets.only(bottom: 16.0),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8.0),
                  child: Image.file(
                    File(_pickedImage!.path),
                    fit: BoxFit.contain,
                  ),
                ),
              )
            else
              Container(
                height: 200,
                margin: const EdgeInsets.only(bottom: 16.0),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: const Center(
                  child: Text(
                    'No image selected',
                    style: TextStyle(color: Colors.grey),
                  ),
                ),
              ),

            // Error message
            if (_errorMessage != null)
              Container(
                margin: const EdgeInsets.only(bottom: 16.0),
                padding: const EdgeInsets.all(12.0),
                decoration: BoxDecoration(
                  color: Colors.red.shade100,
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.red),
                ),
              ),

            // Processing indicator
            if (_isProcessing)
              const Padding(
                padding: EdgeInsets.all(16.0),
                child: Center(
                  child: CircularProgressIndicator(),
                ),
              ),

            // Results
            if (_results.isNotEmpty && !_isProcessing) ...[
              const Text(
                'Detected Text:',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8.0),
              Container(
                padding: const EdgeInsets.all(12.0),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: _results.asMap().entries.map((entry) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4.0),
                      child: Text(
                        '${entry.key + 1}. ${entry.value}',
                        style: const TextStyle(fontSize: 16),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],

            const SizedBox(height: 24.0),

            // Action buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isProcessing ? null : _pickImage,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Pick Image'),
                ),
                ElevatedButton.icon(
                  onPressed: _isProcessing ? null : _captureImage,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}