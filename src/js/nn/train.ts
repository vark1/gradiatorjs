import { Val } from "../Val/val.js";
import { Sequential, Module } from "./nn.js";
import { getStopTraining, endTraining } from "./training_controller.js";
import { calcAccuracy } from "../utils/utils_train.js";

function yieldToBrowser(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0));
}

export async function trainModel(
    model: Sequential,
    X_train: Val,
    Y_train: Val,
    loss_fn: (Y_pred: Val, Y_true: Val) => Val,
    learning_rate: number,
    iterations: number,
    log_frequency: number = 10,
    updateUICallback?: (iter: number, loss: number, accuracy: number) => void
) : Promise<void> {

    console.log(`Starting training.\n iterations: ${iterations}\n alpha: ${learning_rate}\n log frequency: ${log_frequency}`);
    
    const t1 = performance.now();

    try {
        for (let i=0; i<iterations; i++) {
            if (getStopTraining()) {
                console.log(`Training stopped at iteration ${i}`);
                return;
            }

            model.zeroGrad();
            const Y_pred = model.forward(X_train);
            const loss = loss_fn(Y_pred, Y_train);
            if (loss.size !== 1) {
                console.warn(`Loss is not scalar.\n size = ${loss.size}.\n iteration ${i}.\n Skipping backward here`);
                continue;
            }

            loss.backward();

            const params = model.parameters();
            for (const p of params) {
                if (!p.grad || p.data.length !== p.grad.length) {
                    console.warn(`Skipping update for parameter due to missing/mismatched gradient at iteration ${i}. Param shape: ${p.shape}`);
                    continue;
                }

                for (let j=0; j<p.data.length; j++) {
                    if (j < p.grad.length) {
                        p.data[j] -= learning_rate * p.grad[j];
                    }
                }
            }
            
            if ((i % log_frequency) === 0 || i === iterations - 1) {
                const lossValue = loss.data[0];
                const accuracy = calcAccuracy(Y_pred, Y_train);
                console.log(`Iteration ${i}: Loss = ${lossValue.toFixed(6)}, Accuracy = ${accuracy.toFixed(2)}`);
                
                if (updateUICallback) {
                    updateUICallback(i, lossValue, accuracy);
                }
                await yieldToBrowser();
            }
        }
        const t2 = performance.now();
        console.log(`Training done. Time taken: ${((t2-t1)/1000)}`);
        console.log(model)
        
    } catch (error) {
        console.error("Error during training loop execution:", error);
        throw error;
    } finally {
        endTraining();
    }

}