// Compare two face images with ArcFace.
// Usage: node index.js <face-image-1> <face-image-2>
// Any photo works. The largest face in each image is found and aligned automatically.

import { compareFaces } from './arcface.js';

async function main() {
    const [imageA, imageB] = process.argv.slice(2);
    if (!imageA || !imageB) {
        console.log('Usage: node index.js <face-image-1> <face-image-2>');
        process.exit(1);
    }
    const { similarity, threshold, samePerson } = await compareFaces(imageA, imageB);
    console.log(`Similarity: ${similarity.toFixed(4)} (threshold ${threshold})`);
    console.log(samePerson ? 'Result: SAME person' : 'Result: DIFFERENT people');
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
