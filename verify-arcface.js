// Checks that the ArcFace model reproduces the reference output shipped with it.
import { readFileSync } from 'node:fs';
import ort from 'onnxruntime-node';
import { runModel } from './arcface.js';

// The .pb files are ONNX TensorProto messages whose raw float data sits at the end.
const tail = (file, count) => {
    const buf = readFileSync(file);
    const bytes = buf.subarray(buf.length - count * 4);
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
};

const input = tail('models/arcface-test-data/input_0.pb', 3 * 112 * 112);
const expected = tail('models/arcface-test-data/output_0.pb', 512);
const actual = await runModel(new ort.Tensor('float32', input, [1, 3, 112, 112]));

let maxDiff = 0;
for (let i = 0; i < 512; i++) maxDiff = Math.max(maxDiff, Math.abs(actual[i] - expected[i]));
console.log('Output length:', actual.length);
console.log('Largest difference from reference:', maxDiff.toExponential(3));
console.log(maxDiff < 1e-2 ? 'PASS' : 'FAIL');
process.exit(maxDiff < 1e-2 ? 0 : 1);
