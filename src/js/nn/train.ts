import { Val } from "../Val/val.js";
import { Sequential, Module, Dense, Conv } from "./nn.js";
import { getStopTraining, endTraining } from "./training_controller.js";
import { calcAccuracy } from "../utils/utils_train.js";
import { LayerType } from "../utils/utils_vis.js";

function yieldToBrowser(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0));
}

export interface VISActivationData {
    layerIdx: number;
    layerType: LayerType;
    shape: number[];
    activationSample: Float64Array;
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
    
    const overallStartTime = performance.now();
    let totalProcessingTime = 0;
    const t1 = performance.now();

    let sampleX: Val | null = null;

    // creating a sample for visualizer
    if (X_train.size > 0 && X_train.dim === 4) {

        const B = X_train.shape[0];
        const H = X_train.shape[1];
        const W = X_train.shape[2];
        const C = X_train.shape[3];
        
        if (B > 0) { // If there's at least one sample in the batch
            const singleImageFeatureCount = H*W*C;
            const firstImageData = X_train.data.slice(0, singleImageFeatureCount);

            sampleX = new Val([1, H, W, C]);
            sampleX.data = Float64Array.from(firstImageData);
        }
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
                            return {
                                layerIdx: idx,
                                layerType: model.layers[idx] instanceof Conv ? 'conv': 'dense',
                                shape: actVal ? [...actVal.shape] : [],
                                activationSample: actVal.data ? actVal.data : new Float64Array([])
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