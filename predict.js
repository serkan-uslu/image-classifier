const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const jpeg = require('jpeg-js');
const path = require('path');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

function getClassLabels(dirPath) {
    const classLabels = fs.readdirSync(dirPath).filter((item) => {
        return fs.statSync(path.join(dirPath, item)).isDirectory();
    });
    return classLabels;
}

// Usage
const classLabels = getClassLabels('./images');
console.log(classLabels);

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

async function predict(imagePath) {
    // Load the model
    const model = await tf.loadLayersModel('file://./my-model/model.json');

    // Load and preprocess the image
    const image = loadImage(imagePath);

    // Use the model to classify the image
    const prediction = model.predict(image);

    // Because model.predict() returns a tf.Tensor, we need to call one of the tensor's methods (.data(), .array(), etc.)
    // to get a JavaScript TypedArray or a nested array that we can actually use in our code.
    // Here we're using the .dataSync() method to get a TypedArray, then using the Array.from() method to convert it to a regular array.
    const predictionData = Array.from(prediction.dataSync());

    console.log(predictionData); // Output: A list of probabilities for each label

    const predictedClassIndex = predictionData.indexOf(Math.max(...predictionData));
    console.log(`Predicted label: ${classLabels[predictedClassIndex]}`);

}

// Usage
// predict('./images/ayakkabi/ayakkabi1.jpeg');
predict('./ayakkabi.jpeg');
// predict('./elma.jpeg');
// predict('./images/meyve/meyve1.jpeg');
