// Face detection and alignment for ArcFace.
//
// Detector: YuNet from the OpenCV model zoo (models/yunet.onnx).
// Input:  "input", float32 [1, 3, 640, 640], BGR, pixel values 0-255.
// Output: scores, boxes and 5 landmarks per face at strides 8, 16 and 32.
//
// alignFace() turns any photo into the 112x112 aligned face crop ArcFace expects.

import ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DETECTOR_SIZE = 640;
const FACE_SIZE = 112;
const STRIDES = [8, 16, 32];
const DEFAULT_MODEL = path.join(path.dirname(fileURLToPath(import.meta.url)), 'models', 'yunet.onnx');

// Where ArcFace expects the eyes, nose and mouth corners inside a 112x112 crop.
// Order: eye on image left, eye on image right, nose, mouth left, mouth right.
const ARCFACE_TEMPLATE = [
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041],
];

let sessionPromise = null;

function loadDetector(modelPath = DEFAULT_MODEL) {
    if (!sessionPromise) {
        // The detector is tiny, so CPU is fast and avoids GPU start-up cost.
        sessionPromise = ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
        sessionPromise.catch(() => { sessionPromise = null; });
    }
    return sessionPromise;
}

// Decode an image (path or Buffer) to raw RGB pixels, honouring EXIF rotation.
async function decode(image) {
    const { data, info } = await sharp(image).rotate().removeAlpha().toColourspace('srgb').raw()
        .toBuffer({ resolveWithObject: true });
    if (info.channels !== 3) throw new Error(`Expected 3 colour channels, got ${info.channels}`);
    return { data, width: info.width, height: info.height };
}

const rawInput = (img) => ({ raw: { width: img.width, height: img.height, channels: 3 } });

const clamp01 = (v) => Math.min(1, Math.max(0, v));

function iou(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a.width * a.height + b.width * b.height - inter);
}

async function detectInDecoded(img, { minScore = 0.6, nmsThreshold = 0.3 } = {}) {
    // Shrink the photo to fit 640x640, padding the right and bottom with black.
    const ratio = Math.min(DETECTOR_SIZE / img.width, DETECTOR_SIZE / img.height);
    const w = Math.max(1, Math.round(img.width * ratio));
    const h = Math.max(1, Math.round(img.height * ratio));
    const resized = await sharp(img.data, rawInput(img)).resize(w, h, { fit: 'fill' }).raw().toBuffer();

    const plane = DETECTOR_SIZE * DETECTOR_SIZE;
    const input = new Float32Array(3 * plane);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const src = (y * w + x) * 3;
            const dst = y * DETECTOR_SIZE + x;
            input[dst] = resized[src + 2];             // B
            input[plane + dst] = resized[src + 1];     // G
            input[2 * plane + dst] = resized[src];     // R
        }
    }

    const session = await loadDetector();
    const out = await session.run({ input: new ort.Tensor('float32', input, [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]) });

    const faces = [];
    for (const stride of STRIDES) {
        const cols = DETECTOR_SIZE / stride;
        const cls = out[`cls_${stride}`].data;
        const obj = out[`obj_${stride}`].data;
        const bbox = out[`bbox_${stride}`].data;
        const kps = out[`kps_${stride}`].data;
        for (let i = 0; i < cols * cols; i++) {
            const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
            if (score < minScore) continue;
            const col = i % cols;
            const row = Math.floor(i / cols);
            const cx = (col + bbox[i * 4]) * stride;
            const cy = (row + bbox[i * 4 + 1]) * stride;
            const bw = Math.exp(bbox[i * 4 + 2]) * stride;
            const bh = Math.exp(bbox[i * 4 + 3]) * stride;
            const landmarks = [];
            for (let k = 0; k < 5; k++) {
                landmarks.push([
                    (kps[i * 10 + 2 * k] + col) * stride / ratio,
                    (kps[i * 10 + 2 * k + 1] + row) * stride / ratio,
                ]);
            }
            faces.push({
                score,
                box: { x: (cx - bw / 2) / ratio, y: (cy - bh / 2) / ratio, width: bw / ratio, height: bh / ratio },
                landmarks,
            });
        }
    }

    // Drop overlapping duplicates, keeping the highest score.
    faces.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const face of faces) {
        if (kept.every((k) => iou(k.box, face.box) < nmsThreshold)) kept.push(face);
    }
    return kept;
}

// Find all faces in a photo. Each face has a score, a box and 5 landmarks in image pixels.
export async function detectFaces(image, options) {
    return detectInDecoded(await decode(image), options);
}

// Best-fit rotation + uniform scale + shift that moves `from` points onto `to` points.
function similarityTransform(from, to) {
    const n = from.length;
    const mean = (pts, d) => pts.reduce((t, p) => t + p[d], 0) / n;
    const fx = mean(from, 0);
    const fy = mean(from, 1);
    const tx = mean(to, 0);
    const ty = mean(to, 1);
    let num1 = 0;
    let num2 = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
        const x = from[i][0] - fx;
        const y = from[i][1] - fy;
        const u = to[i][0] - tx;
        const v = to[i][1] - ty;
        num1 += x * u + y * v;
        num2 += x * v - y * u;
        den += x * x + y * y;
    }
    const a = num1 / den;
    const b = num2 / den;
    // u = a*x - b*y + c ; v = b*x + a*y + d
    return { a, b, c: tx - (a * fx - b * fy), d: ty - (b * fx + a * fy), scale: Math.hypot(a, b) };
}

// Cut one detected face out of a decoded image as an aligned 112x112 RGB buffer.
async function warpFace(img, face) {
    let src = img;
    let landmarks = face.landmarks;
    let t = similarityTransform(landmarks, ARCFACE_TEMPLATE);

    // If the face is much bigger than 112px, shrink the photo first for a clean result.
    if (t.scale < 0.9) {
        const w = Math.max(1, Math.round(img.width * t.scale));
        const h = Math.max(1, Math.round(img.height * t.scale));
        const data = await sharp(img.data, rawInput(img)).resize(w, h, { fit: 'fill' }).raw().toBuffer();
        src = { data, width: w, height: h };
        landmarks = landmarks.map((p) => [p[0] * w / img.width, p[1] * h / img.height]);
        t = similarityTransform(landmarks, ARCFACE_TEMPLATE);
    }

    const pixel = (x, y, ch) => {
        if (x < 0 || y < 0 || x >= src.width || y >= src.height) return 0;
        return src.data[(y * src.width + x) * 3 + ch];
    };

    // For each output pixel, look up where it comes from in the photo (bilinear sampling).
    const det = t.a * t.a + t.b * t.b;
    const out = Buffer.alloc(FACE_SIZE * FACE_SIZE * 3);
    for (let v = 0; v < FACE_SIZE; v++) {
        for (let u = 0; u < FACE_SIZE; u++) {
            const du = u - t.c;
            const dv = v - t.d;
            const x = (t.a * du + t.b * dv) / det;
            const y = (-t.b * du + t.a * dv) / det;
            const x0 = Math.floor(x);
            const y0 = Math.floor(y);
            const wx = x - x0;
            const wy = y - y0;
            for (let ch = 0; ch < 3; ch++) {
                const value = pixel(x0, y0, ch) * (1 - wx) * (1 - wy)
                    + pixel(x0 + 1, y0, ch) * wx * (1 - wy)
                    + pixel(x0, y0 + 1, ch) * (1 - wx) * wy
                    + pixel(x0 + 1, y0 + 1, ch) * wx * wy;
                out[(v * FACE_SIZE + u) * 3 + ch] = Math.round(value);
            }
        }
    }
    return out;
}

// Photo (path or Buffer) -> aligned 112x112 RGB face crop of the largest face.
// Returns { data, width, height, face, faceCount }. Throws if no face is found.
export async function alignFace(image, options) {
    const img = await decode(image);
    const faces = await detectInDecoded(img, options);
    if (faces.length === 0) throw new Error('No face found in image');
    const area = (f) => f.box.width * f.box.height;
    const face = faces.reduce((best, f) => (area(f) > area(best) ? f : best));
    return { data: await warpFace(img, face), width: FACE_SIZE, height: FACE_SIZE, face, faceCount: faces.length };
}

// Save the aligned crop as an image file so you can see what ArcFace receives.
export async function saveAlignedFace(image, outputPath) {
    const { data } = await alignFace(image);
    await sharp(data, { raw: { width: FACE_SIZE, height: FACE_SIZE, channels: 3 } }).toFile(outputPath);
}
