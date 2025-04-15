import { Val } from "../Val/val.js";
import { Sequential, Module } from "./nn.js";

export function trainModel(
    model: Sequential,
    X_train: Val,
    Y_train: Val,
    loss_fn: (Y_pred: Val, Y_true: Val) => Val,
    learning_rate: number,
    iterations: number,
    log_frequency: number = 100
) : void {

    console.log(`Starting training for ${iterations} iterations with learning rate: ${learning_rate} and log frequency: ${log_frequency}`);
    const t1 = performance.now();

    for (let i=0; i<iterations; i++) {
        // Zero gradients
        model.zeroGrad();

        // Forward pass
        const Y_pred = model.forward(X_train);

        // Compute cost
        const loss = loss_fn(Y_pred, Y_train);

        // backward pass
        loss.backward();

        // update params
        const params = model.parameters();
        for (const p of params) {
            if (!p.grad || p.data.length !== p.grad.length) {
                console.warn(`Skipping update for parameter due to missing/mismatched gradient at iteration ${i}. Param shape: ${p.shape}`);
                continue;
            }

            for (let j=0; j<p.data.length; j++) {
                p.data[i] -= learning_rate * p.grad[i];
            }
        }
        const accuracy = calcAccuracy(Y_pred, Y_train);

        if ((i % log_frequency) === 0 || i === iterations - 1) {
            const lossValue = loss.data[0];
            console.log(`Iteration ${i}: Loss = ${lossValue.toFixed(6)}, Accuracy = ${accuracy.toFixed(2)}`);
        }
    }
    const t2 = performance.now();

    console.log(`Training done. Time taken: ${((t2-t1)/1000)}`);
    console.log(model)
}

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