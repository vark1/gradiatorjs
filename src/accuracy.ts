import { Val } from "./val.js";

export function calcBinaryAccuracy(y_pred_val: Val, y_true_val: Val, threshold: number = 0.5) {
    if(y_pred_val.size !== y_true_val.size) {
        throw new Error(`Cannot cal accuracy: Prediction size ${y_pred_val.size} doesn't match true label size (${y_true_val.size}). Shapes: pred ${y_pred_val.shape}, true ${y_true_val.shape}`);
    }
    
    const n_samples = y_pred_val.size;
    if (n_samples === 0) {
        console.log("Empty input");
        return 100;
    }

    const y_pred: Float64Array = y_pred_val.data;
    const y_true: Float64Array = y_true_val.data;

    let correct_predictions = 0;

    for (let i=0; i<n_samples; i++) {
        const predicted_class = y_pred[i] > threshold ? 1:0;

        const true_label = Math.round(y_true[i]);

        if (predicted_class === true_label) {
            correct_predictions++;
        }
    }

    const accuracy = (correct_predictions / n_samples) * 100;
    return accuracy;
}

export function calcMultiClassAccuracy(y_pred_val: Val, y_true_val: Val) {
    if (y_pred_val.dim !== 2 || y_true_val.dim !== 2 || y_pred_val.shape[0] !== y_true_val.shape[0] || y_pred_val.shape[1] !== y_true_val.shape[1]) {
        throw new Error(`Shape mismatch for multi-class accuracy. Pred: [${y_pred_val.shape.join(',')}], True: [${y_true_val.shape.join(',')}]`);
    }
    
    const batchSize = y_pred_val.shape[0];
    const numClasses = y_pred_val.shape[1];

    if (batchSize === 0) {
        console.log("Empty input");
        return 100.0;
    }
    const y_pred: Float64Array = y_pred_val.data;
    const y_true: Float64Array = y_true_val.data;

    let correct_predictions = 0;

    for (let i=0; i<batchSize; i++) {
        const predOffset = i*numClasses;
        const trueOffset = i*numClasses;

        let maxProb = -1;
        let predicted_class = -1;
        
        // argmax
        for (let j=0; j<numClasses; j++) {
            if (y_pred[predOffset+j]>maxProb) {
                maxProb = y_pred[predOffset+j];
                predicted_class = j;
            }
        }

        let trueClass = -1;
        for (let j=0; j<numClasses; j++) {      // finding true class from one-hot vect
            if (y_true[trueOffset+j] === 1.0) {
                trueClass = j;
                break;
            }
        }
        
        if (predicted_class === trueClass) {
            correct_predictions++;
        }
    }

    const accuracy = (correct_predictions / batchSize) * 100;
    return accuracy;
}