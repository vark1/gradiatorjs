import { Val } from "../Val/val.js";
import * as ops from '../Val/ops.js'

declare const hdf5: any;
declare const pako: any;

async function loadFile(URL: string){
    const response = await fetch(URL);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${URL}: ${response.statusText}`)
    }
    const x = await response.arrayBuffer();
    return new hdf5.File(x, URL.split('/').pop());
}

export async function prepareCatvnoncatData() {
    const testURL = 'http://127.0.0.1:5500/datasets/catvnoncat/test_catvnoncat.h5';
    const trainURL = 'http://127.0.0.1:5500/datasets/catvnoncat/train_catvnoncat.h5';

    const hdf5_train = await loadFile(trainURL);
    const hdf5_test = await loadFile(testURL);
    
    const train_x_raw = hdf5_train.get('train_set_x').value;
    const train_y_raw = hdf5_train.get('train_set_y').value;
    const m_train = train_y_raw.length;

    const train_x = new Val([m_train, 64, 64, 3]);
    train_x.data = Float64Array.from(train_x_raw);

    const train_x_normalized = ops.div(train_x, 255.0);

    const train_y = new Val([m_train, 1]);
    train_y.data = Float64Array.from(train_y_raw);

    const test_x_raw = hdf5_test.get('test_set_x').value;
    const test_y_raw = hdf5_test.get('test_set_y').value;
    const m_test = test_y_raw.length;

    const test_x = new Val([m_test, 64, 64, 3]);
    test_x.data = Float64Array.from(test_x_raw);
    const test_x_normalized = ops.div(test_x, 255.0);
    
    const test_y = new Val([m_test, 1]);
    test_y.data = Float64Array.from(test_y_raw);

    console.log(`# Training examples: ${train_x.shape[0]}`);
    console.log(`# Testing examples: ${test_x.shape[0]}`);
    
    return {
        "train_x_og": train_x,
        "train_x": train_x_normalized,
        "train_y": train_y,
        "test_x_og": test_x,
        "test_x": test_x_normalized,
        "test_y": test_y,
    };
}

// MNIST SPECIFIC
async function loadMNISTData(imagesURL: string, labelsURL: string) {
    async function fetchAndDecompress(URL: string) {
        const response = await fetch(URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${URL}: ${response.statusText}`)
        }
        const compressedBuffer = await response.arrayBuffer();
        const decompressedData = pako.ungzip(new Uint8Array(compressedBuffer));
        return decompressedData
    }

    const imagesBuffer = await fetchAndDecompress(imagesURL);
    const labelsBuffer = await fetchAndDecompress(labelsURL);

    // Parsing Images (IDX3-Ubyte). Using dataview to read binary data
    const imagesDataView = new DataView(imagesBuffer.buffer);

    // MNIST files use big-endian byte order for their 32-bit integers in the header. 
    // setting the littleEndian argument as false
    const numImages = imagesDataView.getUint32(4, false);
    const rows = imagesDataView.getUint32(8, false);
    const cols = imagesDataView.getUint32(12, false);

    // data starts after 16-byte header
    const imageStartByte = 16;
    const imageSize = rows*cols;
    const images = new Float64Array(numImages*imageSize);

    for (let i=0; i<numImages; i++) {
        for (let j=0; j<imageSize; j++) {
            images[i*imageSize+j]=imagesBuffer[imageStartByte+(i*imageSize)+j]/255.0;
        }
    }

    // Parsing Labels (IDX1-Ubyte)
    const labelsDataView = new DataView(labelsBuffer.buffer);
    const numLabels = labelsDataView.getUint32(4, false);

    const labelStartByte = 8;
    const labels = new Uint8Array(numLabels);

    for (let i=0; i<numLabels; i++) {
        labels[i] = labelsBuffer[labelStartByte+i];
    }

    return {
        images: images,
        labels: labels,
        numImages: numImages,
        numRows: rows,
        numCols: cols,
    }
}

export async function prepareMNISTData() {
    const imagesURL = 'http://127.0.0.1:5500/datasets/mnist/gz/train-images-idx3-ubyte.gz';
    const labelsURL = 'http://127.0.0.1:5500/datasets/mnist/gz/train-labels-idx1-ubyte.gz';

    try {
        const data = await loadMNISTData(imagesURL, labelsURL);
        const xTrain = new Val([data.numImages, data.numRows, data.numCols, 1])    // mnist is grayscale so channels=1
        xTrain.data = data.images

        const numClasses = 10 // 10 for one-hot encoding, use 1 if you just want the digit itself
        const yTrain = new Val([data.labels.length, numClasses])
        
        const oneHotLabels = new Float64Array(data.labels.length * numClasses).fill(0)
        for (let i=0; i<data.labels.length; i++) {
            const label = data.labels[i];
            oneHotLabels[i * numClasses + label] = 1.0;
        }
        yTrain.data = oneHotLabels;

        return [xTrain, yTrain];
    } catch (e){
        console.log("ERROR LOADING MNIST DATA: ", e)
    }

    return [new Val([]), new Val([])]
}