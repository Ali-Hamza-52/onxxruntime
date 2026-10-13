// Accuracy test: compares every pair of photos in a folder of labelled faces.
//
// Usage: node test-faces.js [folder] [threshold]
// Folder layout: one sub-folder per person, holding that person's photos.
//   test-faces/Person_A/1.jpg, test-faces/Person_A/2.jpg, test-faces/Person_B/1.jpg ...

import { readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { getEmbedding, cosineSimilarity, DEFAULT_THRESHOLD } from './arcface.js';

const folder = process.argv[2] ?? 'test-faces';
const threshold = Number(process.argv[3] ?? DEFAULT_THRESHOLD);
const IMAGE = /\.(jpe?g|png|webp|bmp)$/i;

// Collect { person, file } for every image.
const photos = [];
for (const person of readdirSync(folder).sort()) {
    const dir = path.join(folder, person);
    if (!statSync(dir).isDirectory()) continue;
    for (const file of readdirSync(dir).sort()) {
        if (IMAGE.test(file)) photos.push({ person, file: path.join(dir, file) });
    }
}
const people = new Set(photos.map((p) => p.person));
console.log(`Found ${photos.length} photos of ${people.size} people in "${folder}"\n`);

// One embedding per photo.
const started = Date.now();
const usable = [];
for (const photo of photos) {
    try {
        photo.embedding = await getEmbedding(photo.file);
        usable.push(photo);
    } catch (err) {
        console.log(`SKIPPED ${photo.file}: ${err.message}`);
    }
}
const seconds = (Date.now() - started) / 1000;
console.log(`Embedded ${usable.length} photos in ${seconds.toFixed(1)}s (${(seconds / usable.length * 1000).toFixed(0)} ms each)\n`);

// Compare every pair.
const same = [];
const different = [];
for (let i = 0; i < usable.length; i++) {
    for (let j = i + 1; j < usable.length; j++) {
        const pair = { a: usable[i], b: usable[j], score: cosineSimilarity(usable[i].embedding, usable[j].embedding) };
        (pair.a.person === pair.b.person ? same : different).push(pair);
    }
}

const stats = (pairs) => {
    const scores = pairs.map((p) => p.score);
    const mean = scores.reduce((t, s) => t + s, 0) / scores.length;
    return `min ${Math.min(...scores).toFixed(3)}  mean ${mean.toFixed(3)}  max ${Math.max(...scores).toFixed(3)}`;
};
const missed = same.filter((p) => p.score < threshold);
const falseMatches = different.filter((p) => p.score >= threshold);
const correct = same.length + different.length - missed.length - falseMatches.length;
const name = (p) => path.relative(folder, p.file);

console.log(`Threshold: ${threshold}`);
console.log(`Same-person pairs:      ${same.length}   scores: ${stats(same)}`);
console.log(`Different-person pairs: ${different.length}   scores: ${stats(different)}`);
console.log('');
console.log(`Missed matches (same person judged different): ${missed.length} of ${same.length}`);
for (const p of missed) console.log(`   ${p.score.toFixed(3)}  ${name(p.a)}  vs  ${name(p.b)}`);
console.log(`False matches (different people judged same):  ${falseMatches.length} of ${different.length}`);
for (const p of falseMatches) console.log(`   ${p.score.toFixed(3)}  ${name(p.a)}  vs  ${name(p.b)}`);
console.log('');
console.log(`Accuracy: ${correct} of ${same.length + different.length} pairs correct (${(correct / (same.length + different.length) * 100).toFixed(2)}%)`);

// Per-person view: lowest score within the person, highest score against anyone else.
console.log('\nPer person:');
for (const person of [...people].sort()) {
    const own = same.filter((p) => p.a.person === person).map((p) => p.score);
    const others = different.filter((p) => p.a.person === person || p.b.person === person).map((p) => p.score);
    if (!own.length || !others.length) continue;
    console.log(`   ${person.padEnd(22)} lowest same ${Math.min(...own).toFixed(3)}   highest other ${Math.max(...others).toFixed(3)}`);
}
