import { Val } from "../Val/val.js";
import { Sequential, Module, Dense, Conv } from "./nn.js";
import { getStopTraining, endTraining } from "./training_controller.js";
import { calcAccuracy } from "../utils/utils_train.js";

function yieldToBrowser(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0));
}

export interface VISActivationData {
    layerIdx: number;
    layerType: 'dense' | 'conv' | 'other';
    shape: number[];
    activationSample: Float64Array | null;
}

export async function trainModel(
    model: Sequential,
    X_train: Val,
    Y_train: Val,
    loss_fn: (Y_pred: Val, Y_true: Val) => Val,
    learning_rate: number,
    iterations: number,
    log_frequency: number = 10,
    vis_frequency: number = 50,
    updateUICallback?: (iter: number, loss: number, accuracy: number, activationData?: VISActivationData[]) => void
) : Promise<void> {

    console.log(`Starting training.\n iterations: ${iterations}\n alpha: ${learning_rate}\n log frequency: ${log_frequency}`);
    
    const t1 = performance.now();

    let sampleX: Val | null = null;
    if (X_train.size > 0 && X_train.dim > 1) {
        const sampleData = X_train.data.slice(0, X_train.shape[1]);
        sampleX = new Val([1, X_train.shape[1]]);
        sampleX.data = sampleData;
    } else if (X_train.size > 0 && X_train.dim === 1) {
        sampleX = X_train.clone();
    }

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
            
            if (((i % log_frequency) === 0 || i === iterations - 1) || (updateUICallback && i % vis_frequency === 0)) {
                const lossValue = loss.data[0];
                const accuracy = calcAccuracy(Y_pred, Y_train);
                let activationVisData: VISActivationData[] = [];

                try{
                    if (updateUICallback && model instanceof Sequential && sampleX) {
                        const allActivations = model.getActivations(sampleX);

                        activationVisData = allActivations.slice(1).map((actVal, idx) => {
                            let layerType: VISActivationData['layerType'] = 'other';
                            if (model.layers[idx] instanceof Dense) {
                                layerType = 'dense';
                            }

                            if (model.layers[idx] instanceof Conv) {
                                layerType = 'conv';
                            }

                            let sampleData: Float64Array | null = null;
                            if (actVal && actVal.size > 0) {
                                sampleData = Float64Array.from(actVal.data);
                            }

                            return {
                                layerIdx: idx,
                                layerType: layerType,
                                shape: actVal ? [...actVal.shape] : [],
                                activationSample: sampleData
                            }
                        })
                    }
                } catch (err: any){
                    console.warn(`could not calculate activations at iteration ${i}: ${err.message}`)
                }

                console.log(`Iteration ${i}: Loss = ${lossValue.toFixed(6)}, Accuracy = ${accuracy.toFixed(2)}`);
                
                if (updateUICallback) {
                    updateUICallback(i, lossValue, accuracy, activationVisData);
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