import 'dart:math' as math;
import 'package:image/image.dart' as img;

/// Sorts polygon points in clockwise order starting from top-left
/// This creates a proper quadrilateral for text detection
List<List<double>> sortPolygon(List<List<double>> points) {
  if (points.length != 4) return points;
  
  // Find the center point
  final centerX = points.map((p) => p[0]).reduce((a, b) => a + b) / 4;
  final centerY = points.map((p) => p[1]).reduce((a, b) => a + b) / 4;
  
  // Sort by angle from center
  final sortedPoints = List<List<double>>.from(points);
  sortedPoints.sort((a, b) {
    final angleA = math.atan2(a[1] - centerY, a[0] - centerX);
    final angleB = math.atan2(b[1] - centerY, b[0] - centerX);
    return angleA.compareTo(angleB);
  });
  
  return sortedPoints;
}

/// Calculates Euclidean distance between two points
double norm(List<double> p1, List<double> p2) {
  double dx = p1[0] - p2[0];
  double dy = p1[1] - p2[1];
  return math.sqrt(dx * dx + dy * dy);
}

/// Crops and warps image based on 4 corner points using perspective transformation
img.Image cropImage(img.Image image, List<List<double>> points) {
  assert(points.length == 4, "shape of points must be 4*2");

  // Calculate crop dimensions
  int cropWidth = math.max(
    norm(points[0], points[1]).toInt(),
    norm(points[2], points[3]).toInt(),
  );

  int cropHeight = math.max(
    norm(points[0], points[3]).toInt(),
    norm(points[1], points[2]).toInt(),
  );

  // Define destination points for perspective transform
  List<List<double>> ptsStd = [
    [0, 0],
    [cropWidth.toDouble(), 0],
    [cropWidth.toDouble(), cropHeight.toDouble()],
    [0, cropHeight.toDouble()],
  ];

  // Apply perspective transformation
  img.Image warped = perspectiveTransform(
    image,
    points,
    ptsStd,
    cropWidth,
    cropHeight,
  );

  // Rotate if height/width ratio >= 1.5
  if (warped.height * 1.0 / warped.width >= 1.5) {
    warped = img.copyRotate(warped, angle: 270);
  }

  return warped;
}

/// Performs perspective transformation on an image
img.Image perspectiveTransform(
  img.Image src,
  List<List<double>> srcPoints,
  List<List<double>> dstPoints,
  int width,
  int height,
) {
  // Calculate perspective transformation matrix
  List<List<double>> matrix = getPerspectiveTransform(srcPoints, dstPoints);

  // Create output image
  img.Image dst = img.Image(width: width, height: height);

  // Apply transformation
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Apply perspective transform to get source coordinates
      double w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
      double srcX = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w;
      double srcY = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w;

      // Bilinear interpolation
      if (srcX >= 0 &&
          srcX < src.width - 1 &&
          srcY >= 0 &&
          srcY < src.height - 1) {
        int x0 = srcX.floor();
        int y0 = srcY.floor();
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double dx = srcX - x0;
        double dy = srcY - y0;

        var p00 = src.getPixel(x0, y0);
        var p10 = src.getPixel(x1, y0);
        var p01 = src.getPixel(x0, y1);
        var p11 = src.getPixel(x1, y1);

        int r =
            ((1 - dx) * (1 - dy) * p00.r +
                    dx * (1 - dy) * p10.r +
                    (1 - dx) * dy * p01.r +
                    dx * dy * p11.r)
                .round();

        int g =
            ((1 - dx) * (1 - dy) * p00.g +
                    dx * (1 - dy) * p10.g +
                    (1 - dx) * dy * p01.g +
                    dx * dy * p11.g)
                .round();

        int b =
            ((1 - dx) * (1 - dy) * p00.b +
                    dx * (1 - dy) * p10.b +
                    (1 - dx) * dy * p01.b +
                    dx * dy * p11.b)
                .round();

        dst.setPixelRgba(x, y, r, g, b, 255);
      }
    }
  }

  return dst;
}

/// Calculates perspective transformation matrix
List<List<double>> getPerspectiveTransform(
  List<List<double>> src,
  List<List<double>> dst,
) {
  // Solve for perspective transformation matrix using least squares
  // This is a simplified version - for production use a proper linear algebra library

  List<List<double>> a = List.generate(8, (_) => List.filled(8, 0.0));
  List<double> b = List.filled(8, 0.0);

  for (int i = 0; i < 4; i++) {
    a[i * 2][0] = src[i][0];
    a[i * 2][1] = src[i][1];
    a[i * 2][2] = 1;
    a[i * 2][6] = -src[i][0] * dst[i][0];
    a[i * 2][7] = -src[i][1] * dst[i][0];
    b[i * 2] = dst[i][0];

    a[i * 2 + 1][3] = src[i][0];
    a[i * 2 + 1][4] = src[i][1];
    a[i * 2 + 1][5] = 1;
    a[i * 2 + 1][6] = -src[i][0] * dst[i][1];
    a[i * 2 + 1][7] = -src[i][1] * dst[i][1];
    b[i * 2 + 1] = dst[i][1];
  }

  List<double> x = solveLinearSystem(a, b);

  return [
    [x[0], x[1], x[2]],
    [x[3], x[4], x[5]],
    [x[6], x[7], 1.0],
  ];
}

/// Solves a linear system using Gaussian elimination
List<double> solveLinearSystem(List<List<double>> a, List<double> b) {
  int n = a.length;

  // Forward elimination
  for (int i = 0; i < n; i++) {
    // Find pivot
    int maxRow = i;
    for (int k = i + 1; k < n; k++) {
      if (a[k][i].abs() > a[maxRow][i].abs()) {
        maxRow = k;
      }
    }

    // Swap rows
    var temp = a[i];
    a[i] = a[maxRow];
    a[maxRow] = temp;

    double tempB = b[i];
    b[i] = b[maxRow];
    b[maxRow] = tempB;

    // Eliminate column
    for (int k = i + 1; k < n; k++) {
      double factor = a[k][i] / a[i][i];
      b[k] -= factor * b[i];
      for (int j = i; j < n; j++) {
        a[k][j] -= factor * a[i][j];
      }
    }
  }

  // Back substitution
  List<double> x = List.filled(n, 0.0);
  for (int i = n - 1; i >= 0; i--) {
    x[i] = b[i];
    for (int j = i + 1; j < n; j++) {
      x[i] -= a[i][j] * x[j];
    }
    x[i] /= a[i][i];
  }

  return x;
}

/// CTC (Connectionist Temporal Classification) Decoder for OCR
class CTCDecoder {
  late List<String> character;

  CTCDecoder() {
    character = [
      'blank',
      '0',
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
      ':',
      ';',
      '<',
      '=',
      '>',
      '?',
      '@',
      'A',
      'B',
      'C',
      'D',
      'E',
      'F',
      'G',
      'H',
      'I',
      'J',
      'K',
      'L',
      'M',
      'N',
      'O',
      'P',
      'Q',
      'R',
      'S',
      'T',
      'U',
      'V',
      'W',
      'X',
      'Y',
      'Z',
      '[',
      '\\',
      ']',
      '^',
      '_',
      '`',
      'a',
      'b',
      'c',
      'd',
      'e',
      'f',
      'g',
      'h',
      'i',
      'j',
      'k',
      'l',
      'm',
      'n',
      'o',
      'p',
      'q',
      'r',
      's',
      't',
      'u',
      'v',
      'w',
      'x',
      'y',
      'z',
      '{',
      '|',
      '}',
      '~',
      '!',
      '"',
      '#',
      '\$',
      '%',
      '&',
      "'",
      '(',
      ')',
      '*',
      '+',
      ',',
      '-',
      '.',
      '/',
      ' ',
      ' ',
    ];
  }

  /// Decodes CTC outputs to text strings
  Map<String, List<dynamic>> call(dynamic outputs) {
    // If outputs is a list, take the last element
    if (outputs is List) {
      outputs = outputs.last;
    }

    // Get argmax indices along axis 2
    List<List<int>> indices = argmax2D(outputs);
    return decode(indices, outputs);
  }

  /// Performs argmax operation on 3D array along axis 2
  List<List<int>> argmax2D(List<List<List<double>>> outputs) {
    List<List<int>> result = [];

    for (var batch in outputs) {
      List<int> batchIndices = [];
      for (var timeStep in batch) {
        int maxIdx = 0;
        double maxVal = timeStep[0];
        for (int i = 1; i < timeStep.length; i++) {
          if (timeStep[i] > maxVal) {
            maxVal = timeStep[i];
            maxIdx = i;
          }
        }
        batchIndices.add(maxIdx);
      }
      result.add(batchIndices);
    }

    return result;
  }

  /// Decodes indices to text strings with confidence scores
  Map<String, List<dynamic>> decode(
    List<List<int>> indices,
    List<List<List<double>>> outputs,
  ) {
    List<String> results = [];
    List<List<double>> confidences = [];
    List<int> ignoredTokens = [0]; // CTC blank token

    for (int i = 0; i < indices.length; i++) {
      // Create selection mask
      List<bool> selection = List.filled(indices[i].length, true);

      // Remove consecutive duplicates
      for (int j = 1; j < indices[i].length; j++) {
        if (indices[i][j] == indices[i][j - 1]) {
          selection[j] = false;
        }
      }

      // Remove ignored tokens (blank)
      for (int j = 0; j < indices[i].length; j++) {
        for (int ignoredToken in ignoredTokens) {
          if (indices[i][j] == ignoredToken) {
            selection[j] = false;
          }
        }
      }

      // Build result string and confidence scores
      StringBuffer result = StringBuffer();
      List<double> confidence = [];

      for (int j = 0; j < indices[i].length; j++) {
        if (selection[j]) {
          result.write(character[indices[i][j]]);
          confidence.add(outputs[i][j][indices[i][j]]);
        }
      }

      results.add(result.toString());
      confidences.add(confidence);
    }

    return {'results': results, 'confidences': confidences};
  }
}
