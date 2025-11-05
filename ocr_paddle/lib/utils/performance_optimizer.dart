import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Performance optimization utilities for OCR processing
class PerformanceOptimizer {
  final int _maxCacheSize = 50;
  final Map<String, dynamic> _cache = {};
  final Queue<String> _cacheOrder = Queue<String>();
  
  Timer? _cleanupTimer;
  
  PerformanceOptimizer() {
    // Periodic cleanup of old cache entries
    _cleanupTimer = Timer.periodic(const Duration(minutes: 5), (_) {
      _cleanupCache();
    });
  }
  
  void dispose() {
    _cleanupTimer?.cancel();
    _cache.clear();
    _cacheOrder.clear();
  }
  
  /// Cache preprocessed images to avoid redundant processing
  img.Image? getCachedImage(String key) {
    return _cache[key] as img.Image?;
  }
  
  void cacheImage(String key, img.Image image) {
    if (_cache.length >= _maxCacheSize) {
      final oldestKey = _cacheOrder.removeFirst();
      _cache.remove(oldestKey);
    }
    
    _cache[key] = image;
    _cacheOrder.add(key);
  }
  
  /// Optimize image for faster processing
  img.Image optimizeImageForOCR(img.Image image) {
    // If image is too large, resize it to reasonable size while maintaining aspect ratio
    if (image.width > 2000 || image.height > 2000) {
      final scale = 2000 / (image.width > image.height ? image.width : image.height);
      final newWidth = (image.width * scale).round();
      final newHeight = (image.height * scale).round();
      
      return img.copyResize(image, width: newWidth, height: newHeight);
    }
    
    return image;
  }
  
  /// Batch process multiple images efficiently
  Future<List<T>> batchProcess<T>(
    List<img.Image> images,
    Future<T> Function(img.Image) processor,
  ) async {
    const batchSize = 4; // Process 4 images at a time
    final results = <T>[];
    
    for (int i = 0; i < images.length; i += batchSize) {
      final endIndex = (i + batchSize < images.length) ? i + batchSize : images.length;
      final batch = images.sublist(i, endIndex);
      
      final futures = batch.map((image) => processor(image)).toList();
      final batchResults = await Future.wait(futures);
      
      results.addAll(batchResults);
    }
    
    return results;
  }
  
  /// Memory-efficient image processing with streaming
  Stream<List<T>> processImagesStream<T>(
    List<img.Image> images,
    Future<T> Function(img.Image) processor,
  ) async* {
    const chunkSize = 8;
    
    for (int i = 0; i < images.length; i += chunkSize) {
      final endIndex = (i + chunkSize < images.length) ? i + chunkSize : images.length;
      final chunk = images.sublist(i, endIndex);
      
      final futures = chunk.map((image) => processor(image)).toList();
      final results = await Future.wait(futures);
      
      yield results;
      
      // Small delay to prevent memory buildup
      await Future.delayed(const Duration(milliseconds: 10));
    }
  }
  
  /// Optimize detection results by filtering low-confidence detections early
  List<List<List<double>>> filterDetectionsEarly(
    List<List<List<double>>> detections, [
    double minConfidence = 0.3,
  ]) {
    return detections.where((box) {
      // Calculate box area as proxy for confidence
      final width = (box[1][0] - box[0][0]).abs();
      final height = (box[3][1] - box[0][1]).abs();
      final area = width * height;
      
      return area >= minConfidence * 1000; // Minimum area threshold
    }).toList();
  }
  
  /// Pre-allocate memory for frequently used operations
  Float32List createPreallocatedBuffer(int size) {
    return Float32List(size);
  }
  
  /// Reuse buffers to reduce garbage collection
  void recycleBuffer(Float32List buffer) {
    // Clear buffer for reuse
    buffer.fillRange(0, buffer.length, 0.0);
  }
  
  /// Measure and report performance metrics
  Map<String, dynamic> getPerformanceMetrics() {
    return {
      'cacheSize': _cache.length,
      'cacheUtilization': _cache.length / _maxCacheSize,
      'timestamp': DateTime.now().toIso8601String(),
    };
  }
  
  void _cleanupCache() {
    final now = DateTime.now();
    final keysToRemove = <String>[];
    
    for (final key in _cache.keys) {
      // Remove entries older than 10 minutes
      if (now.difference(DateTime.parse(key.split('_').last)).inMinutes > 10) {
        keysToRemove.add(key);
      }
    }
    
    for (final key in keysToRemove) {
      _cache.remove(key);
      _cacheOrder.remove(key);
    }
  }
  
  /// Smart preprocessing based on image characteristics
  img.Image smartPreprocess(img.Image image) {
    // Convert to grayscale if it's a color image
    img.Image processed = image;
    
    if (image.numChannels == 3 || image.numChannels == 4) {
      processed = img.grayscale(image);
    }
    
    // Apply thresholding for better text contrast (adaptive threshold not available)
    processed = img.adjustColor(processed, contrast: 1.5);
    
    // Remove noise with Gaussian blur (median filter not available)
    processed = img.gaussianBlur(processed, radius: 1);
    
    return processed;
  }
  
  /// Parallel processing for independent operations
  Future<List<T>> parallelProcess<T>(
    List<T> items,
    Future<T> Function(T) processor, [
    int maxConcurrency = 4,
  ]) async {
    final results = <T>[];
    
    for (final item in items) {
      final future = processor(item);
      results.add(await future);
      
      // Limit concurrency
      if (results.length % maxConcurrency == 0) {
        await Future.delayed(const Duration(milliseconds: 1));
      }
    }
    
    return results;
  }
  
  /// Optimize memory usage by processing in chunks
  Future<List<T>> memoryEfficientProcess<T>(
    List<img.Image> images,
    Future<T> Function(img.Image) processor,
  ) async {
    final results = <T>[];
    
    for (int i = 0; i < images.length; i++) {
      final image = images[i];
      final result = await processor(image);
      results.add(result);
      
      // Force garbage collection periodically
      if (i % 10 == 0) {
        await Future.delayed(const Duration(milliseconds: 5));
      }
    }
    
    return results;
  }
}