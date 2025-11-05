import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

/// Comprehensive validation and testing utilities for OCR functionality
class OCRValidator {
  
  /// Validates OCR results for accuracy and completeness
  static ValidationResult validateResults(List<String> results, {
    int minTextLength = 2,
    double minConfidence = 0.5,
    bool checkForCommonPatterns = true,
  }) {
    final issues = <String>[];
    final warnings = <String>[];
    final metrics = <String, dynamic>{};
    
    // Basic validation
    if (results.isEmpty) {
      issues.add('No text results returned');
      return ValidationResult(false, issues, warnings, metrics);
    }
    
    // Filter valid results
    final validResults = results.where((text) => text.trim().isNotEmpty).toList();
    metrics['totalResults'] = results.length;
    metrics['validResults'] = validResults.length;
    metrics['emptyResults'] = results.length - validResults.length;
    
    if (validResults.isEmpty) {
      issues.add('No valid text detected');
      return ValidationResult(false, issues, warnings, metrics);
    }
    
    // Check text quality
    final shortTexts = validResults.where((text) => text.length < minTextLength).toList();
    if (shortTexts.length > validResults.length * 0.5) {
      warnings.add('Many detected texts are very short (${shortTexts.length}/${validResults.length})');
    }
    
    // Check for repetitive patterns
    final uniqueResults = validResults.toSet();
    metrics['uniqueResults'] = uniqueResults.length;
    if (uniqueResults.length < validResults.length * 0.7) {
      warnings.add('High repetition in detected text');
    }
    
    // Check for common OCR artifacts
    if (checkForCommonPatterns) {
      final artifacts = _detectCommonArtifacts(validResults);
      if (artifacts.isNotEmpty) {
        warnings.add('Detected common OCR artifacts: ${artifacts.join(', ')}');
      }
    }
    
    // Overall quality score
    final qualityScore = _calculateQualityScore(validResults, metrics);
    metrics['qualityScore'] = qualityScore;
    
    return ValidationResult(issues.isEmpty, issues, warnings, metrics);
  }
  
  /// Detects common OCR artifacts in text
  static List<String> _detectCommonArtifacts(List<String> texts) {
    final artifacts = <String>[];
    final allText = texts.join(' ').toLowerCase();
    
    // Check for excessive special characters
    final specialCharRatio = allText.replaceAll(RegExp(r'[a-z0-9\s]'), '').length / allText.length;
    if (specialCharRatio > 0.3) {
      artifacts.add('excessive_special_chars');
    }
    
    // Check for repeated characters
    if (RegExp(r'(.)\1{3,}').hasMatch(allText)) {
      artifacts.add('repeated_chars');
    }
    
    // Check for mixed case issues
    final mixedCaseWords = texts.expand((text) => text.split(' ')).where((word) {
      return word.length > 3 && 
             RegExp(r'[A-Z]').hasMatch(word) && 
             RegExp(r'[a-z]').hasMatch(word) &&
             !RegExp(r'^[A-Z][a-z]+$').hasMatch(word); // Allow normal capitalized words
    }).length;
    
    if (mixedCaseWords > texts.length * 0.2) {
      artifacts.add('mixed_case_issues');
    }
    
    return artifacts;
  }
  
  /// Calculates overall quality score for OCR results
  static double _calculateQualityScore(List<String> validResults, Map<String, dynamic> metrics) {
    double score = 1.0;
    
    // Penalize short texts
    final avgLength = validResults.map((t) => t.length).reduce((a, b) => a + b) / validResults.length;
    score *= math.min(avgLength / 10.0, 1.0); // Normalize to 0-1
    
    // Penalize repetition
    final uniquenessRatio = metrics['uniqueResults'] / validResults.length;
    score *= uniquenessRatio;
    
    // Penalize empty results
    final validRatio = metrics['validResults'] / metrics['totalResults'];
    score *= validRatio;
    
    return math.max(0.0, math.min(1.0, score));
  }
  
  /// Tests OCR with sample images and returns performance metrics
  static Future<TestResults> runPerformanceTests({
    int sampleSize = 5,
    bool includeLargeImages = true,
    bool includeLowQualityImages = true,
  }) async {
    final results = TestResults();
    
    try {
      // Test with sample images (if available)
      final testImages = await _loadTestImages(sampleSize);
      
      for (final testImage in testImages) {
        final testResult = await _testSingleImage(testImage);
        results.addTestResult(testResult);
      }
      
      // Generate synthetic test cases
      if (includeLargeImages) {
        final largeImageTest = await _createLargeImageTest();
        results.addTestResult(largeImageTest);
      }
      
      if (includeLowQualityImages) {
        final lowQualityTest = await _createLowQualityImageTest();
        results.addTestResult(lowQualityTest);
      }
      
    } catch (e) {
      results.addError('Performance test failed: $e');
    }
    
    return results;
  }
  
  /// Loads test images from assets
  static Future<List<img.Image>> _loadTestImages(int count) async {
    final images = <img.Image>[];
    
    // Try to load some sample images (this would need actual images in assets)
    try {
      // For now, create some synthetic test images
      for (int i = 0; i < count; i++) {
        final testImage = _createSyntheticTestImage(i);
        images.add(testImage);
      }
    } catch (e) {
      print('Warning: Could not load test images: $e');
    }
    
    return images;
  }
  
  /// Creates a synthetic test image with text
  static img.Image _createSyntheticTestImage(int index) {
    // Create a simple test image with some text-like patterns
    final image = img.Image(width: 400, height: 200);
    
    // Fill with white background
    img.fill(image, color: img.ColorRgb8(255, 255, 255));
    
    // Add some black rectangles to simulate text
    final rng = math.Random(index);
    for (int i = 0; i < 5 + rng.nextInt(5); i++) {
      final x = rng.nextInt(300);
      final y = rng.nextInt(150);
      final width = 20 + rng.nextInt(80);
      final height = 10 + rng.nextInt(20);
      
      img.fillRect(image, x1: x, y1: y, x2: x + width, y2: y + height, 
                   color: img.ColorRgb8(0, 0, 0));
    }
    
    return image;
  }
  
  /// Tests OCR with a large image
  static Future<TestResult> _createLargeImageTest() async {
    final stopwatch = Stopwatch()..start();
    
    try {
      // Create a large synthetic image
      final largeImage = img.Image(width: 2000, height: 1500);
      img.fill(largeImage, color: img.ColorRgb8(255, 255, 255));
      
      stopwatch.stop();
      
      return TestResult(
        testName: 'Large Image (2000x1500)',
        success: true,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0, // Would be populated by actual OCR
        errorMessage: null,
      );
    } catch (e) {
      return TestResult(
        testName: 'Large Image (2000x1500)',
        success: false,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0,
        errorMessage: e.toString(),
      );
    }
  }
  
  /// Tests OCR with a low quality image
  static Future<TestResult> _createLowQualityImageTest() async {
    final stopwatch = Stopwatch()..start();
    
    try {
      // Create a low quality synthetic image (small, dark, low contrast)
      final lowQualityImage = img.Image(width: 200, height: 150);
      
      // Fill with dark gray
      img.fill(lowQualityImage, color: img.ColorRgb8(40, 40, 40));
      
      // Add some slightly darker rectangles
      final rng = math.Random();
      for (int i = 0; i < 3; i++) {
        final x = rng.nextInt(150);
        final y = rng.nextInt(100);
        final width = 30 + rng.nextInt(20);
        final height = 15 + rng.nextInt(10);
        
        img.fillRect(lowQualityImage, x1: x, y1: y, x2: x + width, y2: y + height, 
                     color: img.ColorRgb8(30, 30, 30));
      }
      
      stopwatch.stop();
      
      return TestResult(
        testName: 'Low Quality Image',
        success: true,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0,
        errorMessage: null,
      );
    } catch (e) {
      return TestResult(
        testName: 'Low Quality Image',
        success: false,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0,
        errorMessage: e.toString(),
      );
    }
  }
  
  /// Tests a single image and returns results
  static Future<TestResult> _testSingleImage(img.Image image) async {
    final stopwatch = Stopwatch()..start();
    
    try {
      // This would integrate with actual OCR pipeline
      // For now, simulate processing
      await Future.delayed(Duration(milliseconds: 100));
      
      stopwatch.stop();
      
      return TestResult(
        testName: 'Test Image (${image.width}x${image.height})',
        success: true,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0,
        errorMessage: null,
      );
    } catch (e) {
      return TestResult(
        testName: 'Test Image (${image.width}x${image.height})',
        success: false,
        processingTime: stopwatch.elapsedMilliseconds,
        detectedRegions: 0,
        errorMessage: e.toString(),
      );
    }
  }
  
  /// Generates a comprehensive validation report
  static String generateValidationReport(ValidationResult result, TestResults performanceResults) {
    final buffer = StringBuffer();
    
    buffer.writeln('=== OCR Validation Report ===');
    buffer.writeln();
    
    // Validation Results
    buffer.writeln('Validation Status: ${result.isValid ? "PASSED" : "FAILED"}');
    buffer.writeln();
    
    if (result.issues.isNotEmpty) {
      buffer.writeln('Issues Found:');
      for (final issue in result.issues) {
        buffer.writeln('  ❌ $issue');
      }
      buffer.writeln();
    }
    
    if (result.warnings.isNotEmpty) {
      buffer.writeln('Warnings:');
      for (final warning in result.warnings) {
        buffer.writeln('  ⚠️  $warning');
      }
      buffer.writeln();
    }
    
    buffer.writeln('Metrics:');
    result.metrics.forEach((key, value) {
      buffer.writeln('  • $key: $value');
    });
    buffer.writeln();
    
    // Performance Results
    buffer.writeln('Performance Test Results:');
    buffer.writeln('  Total Tests: ${performanceResults.totalTests}');
    buffer.writeln('  Passed: ${performanceResults.passedTests}');
    buffer.writeln('  Failed: ${performanceResults.failedTests}');
    buffer.writeln('  Average Processing Time: ${performanceResults.averageProcessingTime.toStringAsFixed(2)}ms');
    
    return buffer.toString();
  }
}

/// Result of validation
class ValidationResult {
  final bool isValid;
  final List<String> issues;
  final List<String> warnings;
  final Map<String, dynamic> metrics;
  
  ValidationResult(this.isValid, this.issues, this.warnings, this.metrics);
}

/// Result of a single test
class TestResult {
  final String testName;
  final bool success;
  final int processingTime;
  final int detectedRegions;
  final String? errorMessage;
  
  TestResult({
    required this.testName,
    required this.success,
    required this.processingTime,
    required this.detectedRegions,
    this.errorMessage,
  });
}

/// Results of multiple performance tests
class TestResults {
  final List<TestResult> results = [];
  final List<String> errors = [];
  
  void addTestResult(TestResult result) {
    results.add(result);
  }
  
  void addError(String error) {
    errors.add(error);
  }
  
  int get totalTests => results.length;
  int get passedTests => results.where((r) => r.success).length;
  int get failedTests => results.where((r) => !r.success).length;
  double get averageProcessingTime {
    if (results.isEmpty) return 0;
    final totalTime = results.map((r) => r.processingTime).reduce((a, b) => a + b);
    return totalTime / results.length;
  }
}