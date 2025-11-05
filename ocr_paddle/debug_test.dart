import 'dart:io';
import 'package:image/image.dart' as img;
import 'lib/utils/image_splitter.dart';

void main() {
  final splitter = ImageSplitter();
  
  // Test with good quality image
  final goodImage = img.Image(width: 400, height: 300);
  img.fill(goodImage, color: img.ColorRgb8(200, 200, 200)); // Light gray
  
  print('Image dimensions: ${goodImage.width}x${goodImage.height}');
  
  // Calculate brightness manually
  final grayscale = goodImage.numChannels == 3 ? img.grayscale(goodImage) : goodImage;
  double sum = 0;
  
  for (int y = 0; y < grayscale.height; y++) {
    for (int x = 0; x < grayscale.width; x++) {
      sum += grayscale.getPixel(x, y).r;
    }
  }
  
  final avgBrightness = sum / (grayscale.width * grayscale.height);
  print('Average brightness: $avgBrightness');
  
  // Calculate contrast manually
  final pixels = <int>[];
  
  for (int y = 0; y < grayscale.height; y++) {
    for (int x = 0; x < grayscale.width; x++) {
      pixels.add(grayscale.getPixel(x, y).r.toInt());
    }
  }
  
  pixels.sort();
  final q1 = pixels[pixels.length ~/ 4];
  final q3 = pixels[3 * pixels.length ~/ 4];
  
  final contrast = (q3 - q1).toDouble();
  print('Contrast: $contrast');
  
  print('Validation result: ${splitter.validateImage(goodImage)}');
  
  // Test with dark image
  final darkImage = img.Image(width: 400, height: 300);
  img.fill(darkImage, color: img.ColorRgb8(10, 10, 10)); // Very dark
  
  final darkBrightness = _calculateAverageBrightness(darkImage);
  final darkContrast = _calculateContrast(darkImage);
  print('Dark image - Brightness: $darkBrightness, Contrast: $darkContrast');
  print('Dark image validation: ${splitter.validateImage(darkImage)}');
}

double _calculateAverageBrightness(img.Image image) {
  final grayscale = image.numChannels == 3 ? img.grayscale(image) : image;
  double sum = 0;
  
  for (int y = 0; y < grayscale.height; y++) {
    for (int x = 0; x < grayscale.width; x++) {
      sum += grayscale.getPixel(x, y).r;
    }
  }
  
  return sum / (grayscale.width * grayscale.height);
}

double _calculateContrast(img.Image image) {
  final grayscale = image.numChannels == 3 ? img.grayscale(image) : image;
  final pixels = <int>[];
  
  for (int y = 0; y < grayscale.height; y++) {
    for (int x = 0; x < grayscale.width; x++) {
      pixels.add(grayscale.getPixel(x, y).r.toInt());
    }
  }
  
  pixels.sort();
  final q1 = pixels[pixels.length ~/ 4];
  final q3 = pixels[3 * pixels.length ~/ 4];
  
  return (q3 - q1).toDouble();
}