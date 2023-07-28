const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const jpeg = require('jpeg-js');
const path = require('path');

const IMAGE_WIDTH = 28;  // Need to resize images to this width.
const IMAGE_HEIGHT = 28;  // Need to resize images to this height.
const IMAGE_CHANNELS = 3;  // RGB images

function loadImage(filePath) {
    const buf = fs.readFileSync(filePath);
    const pixels = jpeg.decode(buf, true);
    return tf.tidy(() => {
        // Convert image to tensor
        let img = tf.node.decodeImage(buf);
        // Normalize values to range 0 - 1.
        img = tf.cast(img, 'float32').div(tf.scalar(255));
        // Resize to expected size
        img = tf.image.resizeBilinear(img, [IMAGE_WIDTH, IMAGE_HEIGHT]);
        // Reshape tensor to fit into the model
        return img.expandDims(0);
    });
}

function loadImagesFromDir(dirPath) {
    const images = [];
    const labels = [];
    const classes = fs.readdirSync(dirPath);
    let label = 0;
    for (let i = 0; i < classes.length; i++) {
        const classDir = path.join(dirPath, classes[i]);
        const imagePaths = fs.readdirSync(classDir).map((f) => path.join(classDir, f));
        for (const imagePath of imagePaths) {
            images.push(loadImage(imagePath));
            labels.push(label);
        }
        label++;
    }
    return { images, labels };
}

const { images, labels } = loadImagesFromDir('./images');
const xs = tf.concat(images);
const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), labels.length);


const model = tf.sequential();
model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    filters: 32,
    kernelSize: 3,
    activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));


model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});


async function trainAndSave() {
    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 16,
        learningRate: 0.001
    });

    // Save the model
    await model.save('file://./my-model');
}

trainAndSave();
