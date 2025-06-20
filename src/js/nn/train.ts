import { Val } from "../Val/val.js";
import { Sequential } from "./nn.js";
import { getStopTraining, endTraining } from "./training_controller.js";
import { calcBinaryAccuracy, calcMultiClassAccuracy } from "../utils/utils_train.js";
import { visPackage } from "../types_and_interfaces/vis_interfaces.js";
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
    multiClass: boolean,
    updateUICallback: (
        epoch: number,
        batch_idx: number,
        iter: number, 
        loss: number, 
        accuracy: number, 
    ) => void,
    updateActivationVis: (model: Sequential, visdata: visPackage)=> void
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
                    console.log("Y_pred", Y_pred)
                    console.log("Y_batch", Y_batch)
                    let accuracy = 100;
                    if (multiClass) {
                        accuracy = calcMultiClassAccuracy(Y_pred, Y_batch);
                    } else {
                        accuracy = calcBinaryAccuracy(Y_pred, Y_batch);
                    }
                    updateUICallback(e, batch_idx, loss.data[0], accuracy, iterTime);
                }

                if (batch_idx % vis_freq === 0) {
                    assert(model instanceof Sequential, ()=>`Model is not an instance of sequential.`)

                    const sampleX = new Val(batch.x.shape.slice(1)).reshape([1, ...batch.x.shape.slice(1)]);
                    sampleX.data = batch.x.data.slice(0, batch.x.size / batch.x.shape[0]);

                    let sampleY_label = -1;
                    const y_features = batch.y.size / batch.y.shape[0];
                    if (y_features === 1) { // If label is a single number
                        sampleY_label = batch.y.data[0];
                    } else { // If label is one-hot encoded
                        const y_sample_data = batch.y.data.slice(0, y_features);
                        sampleY_label = y_sample_data.indexOf(1);
                    }

                    const layerOutputs = model.getLayerOutputs(sampleX);

                    const visData = {
                        sampleX: sampleX,
                        sampleY_label: sampleY_label,
                        layerOutputs: layerOutputs
                    };
                    updateActivationVis(model, visData);
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