// const express = require('express');

// const app = express();

// app.get('/', (req, res) => {
//     res.send('Hello from Node.js server!');
// });

// const port = 3333;
// app.listen(port, () => {
//     console.log(`Server listening on port ${port}`);
// })

import ort from 'onnxruntime-node'

// Create a session on the GPU through DirectML, falling back to CPU if that fails.
async function createSession(modelPath) {
    try {
        const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['dml', 'cpu'] });
        console.log('Execution provider: DirectML (GPU), CPU fallback');
        return session;
    } catch (err) {
        console.warn(`DirectML unavailable (${err.message}). Using CPU.`);
        const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
        console.log('Execution provider: CPU');
        return session;
    }
}

// Build a 28x28 grayscale image of a vertical stroke, which looks like the digit 1.
function makeDigitOneImage() {
    const size = 28;
    const pixels = new Float32Array(size * size); // 0 = black background
    for (let row = 4; row < 24; row++) {
        for (let col = 13; col < 16; col++) {
            pixels[row * size + col] = 1.0; // white stroke
        }
    }
    return pixels;
}

function softmax(values) {
    const max = Math.max(...values);
    const exps = values.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
}

async function main() {
    // Load the MNIST handwritten digit classifier
    const session = await createSession('models/mnist.onnx');
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    console.log(`Model input: "${inputName}", output: "${outputName}"`);

    // Prepare input data: one image, one channel, 28x28 pixels
    const input = new ort.Tensor('float32', makeDigitOneImage(), [1, 1, 28, 28]);

    // Run the model
    const output = await session.run({ [inputName]: input });

    // Print the output: one score per digit 0-9
    const probabilities = softmax(Array.from(output[outputName].data));
    const predicted = probabilities.indexOf(Math.max(...probabilities));
    probabilities.forEach((p, digit) => console.log(`  ${digit}: ${(p * 100).toFixed(2)}%`));
    console.log(`Predicted digit: ${predicted}`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
