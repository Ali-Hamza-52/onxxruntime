// ArcFace face recognition helpers built on ONNX Runtime.
//
// Model: ArcFace ResNet100 from the ONNX model zoo (models/arcface.onnx).
// Input:  "data", float32 [1, 3, 112, 112], RGB, pixel values 0-255, channel-first.
// Output: "fc1",  float32 [1, 512], the face embedding.
//
// ArcFace needs a cropped, aligned face. getEmbedding() handles that for you:
// it finds the largest face in the photo and aligns it (see face-detector.js).

import ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { alignFace } from './face-detector.js';

const FACE_SIZE = 112;
const DEFAULT_MODEL = path.join(path.dirname(fileURLToPath(import.meta.url)), 'models', 'arcface.onnx');

// Cosine similarity above this value is treated as the same person.
// Tune it on your own data: raise it for fewer false matches, lower it for fewer misses.
export const DEFAULT_THRESHOLD = 0.4;

let sessionPromise = null;

// Load the model once and reuse it. Tries DirectML (GPU) first, then CPU.
export function loadArcFace(modelPath = DEFAULT_MODEL) {
    if (!sessionPromise) {
        sessionPromise = (async () => {
            try {
                return await ort.InferenceSession.create(modelPath, { executionProviders: ['dml', 'cpu'] });
            } catch (err) {
                console.warn(`DirectML unavailable (${err.message}). Using CPU.`);
                return await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
            }
        })();
        sessionPromise.catch(() => { sessionPromise = null; });
    }
    return sessionPromise;
}

// Convert interleaved RGB pixels (RGBRGB...) of a 112x112 face into the model's tensor.
function toTensor(rgb) {
    // The model wants all R values, then all G, then all B.
    const plane = FACE_SIZE * FACE_SIZE;
    const chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
        chw[i] = rgb[i * 3];
        chw[plane + i] = rgb[i * 3 + 1];
        chw[2 * plane + i] = rgb[i * 3 + 2];
    }
    return new ort.Tensor('float32', chw, [1, 3, FACE_SIZE, FACE_SIZE]);
}

// Convert an image (file path or Buffer) into the tensor ArcFace expects.
// By default the face is detected and aligned automatically.
// Pass { aligned: true } only if the image is already a tight, aligned face crop.
export async function preprocess(image, { aligned = false } = {}) {
    if (!aligned) return toTensor((await alignFace(image)).data);

    const { data, info } = await sharp(image)
        .rotate() // honour EXIF orientation
        .resize(FACE_SIZE, FACE_SIZE, { fit: 'cover' })
        .removeAlpha()
        .toColourspace('srgb')
        .raw()
        .toBuffer({ resolveWithObject: true });
    if (info.channels !== 3) throw new Error(`Expected 3 colour channels, got ${info.channels}`);
    return toTensor(data);
}

// Scale a vector to length 1 so that a dot product equals cosine similarity.
export function normalize(vector) {
    let sum = 0;
    for (const v of vector) sum += v * v;
    const norm = Math.sqrt(sum) || 1;
    return Float32Array.from(vector, (v) => v / norm);
}

// Run the model on a ready-made tensor and return the raw 512 values.
export async function runModel(tensor) {
    const session = await loadArcFace();
    const result = await session.run({ [session.inputNames[0]]: tensor });
    return result[session.outputNames[0]].data;
}

// Photo (path or Buffer) -> normalised 512-number embedding. Store this per person.
// Throws "No face found in image" if the photo has no detectable face.
export async function getEmbedding(image, options) {
    return normalize(await runModel(await preprocess(image, options)));
}

// Similarity of two normalised embeddings, from -1 to 1. Higher means more alike.
export function cosineSimilarity(a, b) {
    if (a.length !== b.length) throw new Error('Embeddings have different lengths');
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
}

// Compare two photos and decide whether they show the same person.
export async function compareFaces(imageA, imageB, threshold = DEFAULT_THRESHOLD, options) {
    const a = await getEmbedding(imageA, options);
    const b = await getEmbedding(imageB, options);
    const similarity = cosineSimilarity(a, b);
    return { similarity, threshold, samePerson: similarity >= threshold };
}
