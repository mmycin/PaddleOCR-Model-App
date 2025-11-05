import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import '../lib/utils/image_splitter.dart';
import '../lib/utils/ocr_validator.dart';
import '../lib/utils/utils.dart';
import '../lib/nets/detection.dart';

void main() {
  group('Image Text Detection Tests', () {
    late ImageSplitter splitter;
    
    setUp(() {
      splitter = ImageSplitter();
    });

    group('Image Splitter Tests', () {
      test('should not split small images', () {
        final smallImage = img.Image(width: 500, height: 500);
        final tiles = splitter.splitImage(smallImage);
        
        expect(tiles.length, 1);
        expect(tiles[0].width, 500);
        expect(tiles[0].height, 500);
      });

      test('should split large images into multiple tiles', () {
        final largeImage = img.Image(width: 2000, height: 1500);
        final tiles = splitter.splitImage(largeImage);
        
        expect(tiles.length, greaterThan(1));
        
        // Each tile should not exceed maxTileSize
        for (final tile in tiles) {
          expect(tile.width, lessThanOrEqualTo(splitter.maxTileSize));
          expect(tile.height, lessThanOrEqualTo(splitter.maxTileSize));
        }
      });

      test('should handle edge case of very large image', () {
        final hugeImage = img.Image(width: 5000, height: 5000);
        final tiles = splitter.splitImage(hugeImage);
        
        expect(tiles.length, greaterThan(4));
        expect(tiles.length, lessThan(50)); // Reasonable number of tiles
      });

      test('should validate image quality correctly', () {
        // Test with good quality image
        final goodImage = img.Image(width: 400, height: 300);
        img.fill(goodImage, color: img.ColorRgb8(200, 200, 200)); // Light gray
        
        expect(splitter.validateImage(goodImage), isTrue);
        
        // Test with too small image
        final smallImage = img.Image(width: 5, height: 5);
        expect(splitter.validateImage(smallImage), isFalse);
        
        // Test with too dark image
        final darkImage = img.Image(width: 400, height: 300);
        img.fill(darkImage, color: img.ColorRgb8(10, 10, 10)); // Very dark
        
        expect(splitter.validateImage(darkImage), isFalse);
      });

      test('should merge detections correctly', () {
        // Create mock detection results from tiles
        final tileDetections = [
          [ // Tile 1 detections
            [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]], // Box in tile 1
          ],
          [ // Tile 2 detections  
            [[60.0, 10.0], [100.0, 10.0], [100.0, 30.0], [60.0, 30.0]], // Box in tile 2
          ],
        ];
        
        final tilePositions = [
          {'x': 0, 'y': 0},    // Tile 1 at origin
          {'x': 80, 'y': 0},   // Tile 2 offset by 80px
        ];
        
        final merged = splitter.mergeDetections(
          tileDetections,
          tilePositions,
          200, // Original image width
          100, // Original image height
        );
        
        expect(merged.length, 2);
        
        // Verify coordinates are adjusted correctly
        expect(merged[0][0][0], 10); // First box unchanged (origin tile)
        expect(merged[1][0][0], 140); // Second box offset by tile position
      });

      test('should remove duplicate boxes correctly', () {
        // Create overlapping boxes
        final overlappingBoxes = [
          [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]],     // Large box
          [[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]],   // Smaller box inside
          [[200.0, 200.0], [300.0, 200.0], [300.0, 250.0], [200.0, 250.0]], // Non-overlapping box
        ];
        
        // This is testing the private method indirectly through mergeDetections
        final result = splitter.mergeDetections(
          [overlappingBoxes],
          [{'x': 0, 'y': 0}],
          400,
          400,
        );
        
        // Should keep the larger box and the non-overlapping one
        expect(result.length, lessThanOrEqualTo(2));
      });
    });

    group('OCR Validator Tests', () {
      test('should validate good OCR results', () {
        final goodResults = [
          'Hello World',
          'Flutter OCR',
          'Text Detection',
          'Machine Learning',
        ];
        
        final validation = OCRValidator.validateResults(goodResults);
        
        expect(validation.isValid, isTrue);
        expect(validation.issues, isEmpty);
        expect(validation.metrics['totalResults'], 4);
        expect(validation.metrics['validResults'], 4);
      });

      test('should detect empty results', () {
        final emptyResults = <String>[];
        
        final validation = OCRValidator.validateResults(emptyResults);
        
        expect(validation.isValid, isFalse);
        expect(validation.issues, contains('No text results returned'));
      });

      test('should detect low quality results', () {
        final poorResults = [
          '',
          '',
          '',
          '',
        ];
        
        final validation = OCRValidator.validateResults(poorResults);
        
        expect(validation.isValid, isFalse);
        expect(validation.issues, contains('No valid text detected'));
      });

      test('should detect repetitive results', () {
        final repetitiveResults = [
          'Same',
          'Same',
          'Same',
          'Same',
        ];
        
        final validation = OCRValidator.validateResults(repetitiveResults);
        
        expect(validation.warnings, isNotEmpty);
        expect(validation.warnings.any((w) => w.contains('repetition')), isTrue);
      });

      test('should detect common OCR artifacts', () {
        final resultsWithArtifacts = [
          'H3ll0 W0rld!!!',
          'T3xt D3t3ct10n',
          'Mach1ne Learn1ng',
        ];
        
        final validation = OCRValidator.validateResults(resultsWithArtifacts);
        
        expect(validation.warnings, isNotEmpty);
        expect(validation.warnings.any((w) => w.contains('artifacts')), isTrue);
      });

      test('should calculate quality score correctly', () {
        final mixedResults = [
          'Good quality text here',
          '',
          'Another good text',
          'Short',
        ];
        
        final validation = OCRValidator.validateResults(mixedResults);
        
        expect(validation.metrics, contains('qualityScore'));
        final qualityScore = validation.metrics['qualityScore'] as double;
        expect(qualityScore, greaterThan(0.0));
        expect(qualityScore, lessThanOrEqualTo(1.0));
      });
    });

    group('Utility Function Tests', () {
      test('should sort polygon points correctly', () {
        final unsortedPoints = [
          [100.0, 50.0],  // Should be sorted to position 2
          [0.0, 0.0],     // Should be first
          [100.0, 0.0],   // Should be second
          [0.0, 50.0],    // Should be last
        ];
        
        final sorted = sortPolygon(unsortedPoints);
        
        expect(sorted[0], [0.0, 0.0]);     // Top-left
        expect(sorted[1], [100.0, 0.0]);   // Top-right
        expect(sorted[2], [100.0, 50.0]);  // Bottom-right
        expect(sorted[3], [0.0, 50.0]);    // Bottom-left
      });

      test('should handle degenerate polygons', () {
        final degeneratePoints = [
          [0.0, 0.0],
          [0.0, 0.0], // Duplicate point
          [10.0, 10.0],
          [10.0, 10.0], // Another duplicate
        ];
        
        final sorted = sortPolygon(degeneratePoints);
        
        expect(sorted.length, 4); // Should still return 4 points
      });

      test('should calculate Euclidean distance correctly', () {
        final point1 = [0.0, 0.0];
        final point2 = [3.0, 4.0];
        
        final distance = norm(point1, point2);
        
        expect(distance, closeTo(5.0, 0.001)); // 3-4-5 triangle
      });
    });

    group('Error Handling Tests', () {
      test('should handle null image gracefully', () {
        expect(() => splitter.validateImage(null as dynamic), throwsA(isA<Error>()));
      });

      test('should handle empty detection results', () {
        final emptyDetections = <List<List<List<double>>>>[];
        final positions = <Map<String, int>>[];
        
        final merged = splitter.mergeDetections(emptyDetections, positions, 100, 100);
        
        expect(merged, isEmpty);
      });

      test('should handle invalid box coordinates', () {
        final invalidBoxes = [
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], // Zero area
          [[10.0, 10.0], [5.0, 10.0], [10.0, 5.0], [5.0, 5.0]], // Self-intersecting
        ];
        
        // Should handle gracefully without crashing
        final result = splitter.mergeDetections([invalidBoxes], [{'x': 0, 'y': 0}], 100, 100);
        
        expect(result, isNotNull);
      });
    });

    group('Performance Tests', () {
      test('should process multiple tiles efficiently', () {
        final stopwatch = Stopwatch()..start();
        
        final largeImage = img.Image(width: 3000, height: 2000);
        final tiles = splitter.splitImage(largeImage);
        
        stopwatch.stop();
        
        expect(tiles.length, greaterThan(1));
        expect(stopwatch.elapsedMilliseconds, lessThan(1000)); // Should be fast
      });

      test('should handle high overlap scenarios', () {
        final highOverlapSplitter = ImageSplitter(overlapRatio: 0.5);
        
        final image = img.Image(width: 1500, height: 1000);
        final tiles = highOverlapSplitter.splitImage(image);
        
        // With 50% overlap, should create more tiles
        expect(tiles.length, greaterThan(4));
      });
    });
  });
}