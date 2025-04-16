import { Val } from "../Val/val.js";

export function calcAccuracy(
    y_pred_val: Val,
    y_true_val: Val,
    threshold: number = 0.5
) {
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