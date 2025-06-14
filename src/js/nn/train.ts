import { Val } from "../Val/val.js";
import { Sequential, Dense, Conv, MaxPool2D, Flatten } from "./nn.js";
import { getStopTraining, endTraining } from "./training_controller.js";
import { calcAccuracy } from "../utils/utils_train.js";
import { VISActivationData } from "../types_and_interfaces/vis_interfaces.js";
import { assert } from "../utils/utils.js";

function yieldToBrowser(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0));
}

function createBatchVal(ogVal: Val, batchIndices: number[], currentBatchSize: number, features: number) {
    const batchShape = [currentBatchSize, ...ogVal.shape.slice(1)];
    const batchVal = new Val(batchShape);

    for (let k=0; k<currentBatchSize; k++) {
        const ogIdx = batchIndices[k];
        const sourceOffset = ogIdx*features;
        const destOffset = k*features;
        batchVal.data.set(ogVal.data.subarray(sourceOffset, sourceOffset+features), destOffset);
    }
    return batchVal;
}

// generator that will yield mini-batches from the full dataset 
function* getMiniBatch(X: Val, Y: Val, batchSize: number){
    const numSamples = X.shape[0]
    const indices = Array.from({ length: numSamples }, (_, i) => i);

    const xFeatures = X.size / numSamples;  // H*W*C for images, or F for dense
    const yFeatures = Y.size / numSamples;  // num classes for 1-hot labels

    for (let i=0; i<numSamples; i+=batchSize) {
        const batchIndices = indices.slice(i, i+batchSize);
        const currentBatchSize = batchIndices.length;

        const xBatchVal = createBatchVal(X, batchIndices, currentBatchSize, xFeatures)
        const yBatchVal = createBatchVal(Y, batchIndices, currentBatchSize, yFeatures)

        yield {x: xBatchVal, y: yBatchVal}
    }
}

export async function trainModel(
    model: Sequential,
    X_train: Val,
    Y_train: Val,
    loss_fn: (Y_pred: Val, Y_true: Val) => Val,
    l_rate: number,
    epochs: number,
    batch_size: number,
    update_ui_freq: number = 10,
    vis_freq: number = 50,
    updateUICallback: (
        epoch: number,
        batch_idx: number,
        iter: number, 
        loss: number, 
        accuracy: number, 
    ) => void,
    updateActivationVis: (actvisdata: VISActivationData[], model: Sequential, sampleX: Val)=> void
) : Promise<void> {

    console.log(`----starting training. ${epochs} epochs, batch size ${batch_size}----`);
    
    let totalProcessingTime = 0;
    let iteration = 0;

    try {
        for (let e=0; e<epochs; e++) {
            
            const batchGenerator = getMiniBatch(X_train, Y_train, batch_size)
            let batch_idx = 0;
            
            for (const batch of batchGenerator) {
                if (getStopTraining()) {
                    console.log(`Training stopped at epoch ${e}`);
                    return;
                }

                const iterStartTime = performance.now();
                const { x: X_batch, y: Y_batch } = batch;

                model.zeroGrad();
                const Y_pred = model.forward(X_batch);
                const loss = loss_fn(Y_pred, Y_batch);
                loss.backward();

                const params = model.parameters();
                
                for (const p of params) {
                    if (!p.grad || p.data.length !== p.grad.length) {
                        console.warn(`Skipping update for parameter due to missing/mismatched gradient at iteration ${e}. Param shape: ${p.shape}`);
                        continue;
                    }

                    for (let j=0; j<p.data.length; j++) {
                        if (j < p.grad.length) {
                            p.data[j] -= l_rate * p.grad[j];
                        }
                    }
                }

                const iterEndTime = performance.now();
                const iterTime = iterEndTime - iterStartTime;
                totalProcessingTime += iterTime;
                iteration++;

                if (batch_idx % update_ui_freq === 0) {
                    const accuracy = calcAccuracy(Y_pred, Y_batch);
                    updateUICallback(e, batch_idx, loss.data[0], accuracy, iterTime);
                }

                if (batch_idx % vis_freq === 0) {
                    let activationVisData: VISActivationData[] = [];

                    assert(model instanceof Sequential, ()=>`Model is not an instance of sequential.`)
                    const sampleX_for_vis = new Val(X_batch.shape.slice(1)).reshape([1, ...X_batch.shape.slice(1)])
                    sampleX_for_vis.data = X_batch.data.slice(0, X_batch.size / X_batch.shape[0]);

                    const layerOutputs = model.getLayerOutputs(sampleX_for_vis);

                    activationVisData = model.layers.map((engineLayer, layerModelIdx) => {
                        const output = layerOutputs[layerModelIdx];
                        let layerType: VISActivationData['layerType'] = 'dense';
                        let zVal: Val | null = output.Z;
                        let aVal: Val | null = output.A;

                        if (engineLayer instanceof Dense) {
                            layerType = 'dense';
                        } else if (engineLayer instanceof Conv) {
                            layerType = 'conv';
                        } else if (engineLayer instanceof MaxPool2D) {
                            layerType = 'maxpool';
                            zVal = aVal;
                        } else if (engineLayer instanceof Flatten) {
                            layerType = 'flatten';
                            zVal = aVal;
                        }

                        return {
                            layerIdx: layerModelIdx,
                            layerType: layerType,
                            zShape: zVal ? [...zVal.shape] : [],
                            aShape: aVal ? [...aVal.shape] : [],
                            // zSample: zVal ? Float64Array.from(zVal.data) : null,
                            zSample: zVal ? zVal : null,
                            // aSample: aVal ? Float64Array.from(aVal.data) : null,                            
                            aSample: aVal ? aVal : null,
                        }
                    })
                    updateActivationVis(activationVisData, model, sampleX_for_vis);
                }
                batch_idx++;
                await yieldToBrowser();
            }
        }

        console.log(`Training finished.`);
        console.log(`Total processing time: ${(totalProcessingTime / 1000).toFixed(3)}s over ${iteration} iterations.`);
        console.log(`Average time per batch: ${(totalProcessingTime / iteration / 1000).toFixed(4)}s`);
        console.log(model)
        
    } catch (error) {
        console.error("Error during training loop execution:", error);
        throw error;
    } finally {
        endTraining();
    }
}