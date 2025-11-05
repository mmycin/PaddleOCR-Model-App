# Tensor Creation API Fix Needed

The `OrtValueTensor.createTensorWithDataAsFloat32List` method doesn't exist in the current onnxruntime package.

## To Fix:

You need to check the actual API for creating tensors from Float32List in your onnxruntime package version (1.4.1).

Common patterns to try:

1. **Constructor pattern:**
   ```dart
   return ort.OrtValueTensor(shape, buffer);
   ```

2. **OrtEnv method:**
   ```dart
   return ort.OrtEnv.instance.createTensorValue(shape, buffer);
   ```

3. **Factory method:**
   ```dart
   return ort.OrtValueTensor.fromList(shape, buffer);
   ```

4. **Direct data passing:**
   ```dart
   // Pass Float32List directly to session.run
   _session.run(runOpts, {_inputName: buffer});
   ```

## Files that need fixing:
- `lib/nets/classification.dart` - line 123
- `lib/nets/detection.dart` - line 131  
- `lib/nets/recognition.dart` - line 155

Check the onnxruntime package documentation or examples for the correct API.
