import { Val } from "./val.js";
import { Sequential } from "./layers.js";
import { getStopTraining, endTraining, getIsPaused } from "./state_management.js";
import { calcBinaryAccuracy, calcMultiClassAccuracy } from "./accuracy.js";
import { TrainingProgress } from "./js/types_and_interfaces/vis_interfaces.js";
import { assert } from "./utils.js";
import { NetworkParams } from "types_and_interfaces/general.js";

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
function* getMiniBatch(X: Val, Y: Val, batchSize: number, shuffle: boolean = true){
    const numSamples = X.shape[0]
    const indices = Array.from({ length: numSamples }, (_, i) => i);

    if (shuffle) {
        for (let i=indices.length-1; i>0; i--) {
            const j=Math.floor(Math.random()*(i+1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
    }

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
    params: NetworkParams,
    callbacks?: {
        onBatchEnd?: (progress: TrainingProgress) => void;
    }
) : Promise<void> {

    const {loss_fn, l_rate, epochs, batch_size, multiClass} = params
    console.log(`----starting training. ${epochs} epochs, batch size ${batch_size}----`);
    
    let totalProcessingTime = 0;
    let iteration = 0;

    try {
        for (let e=0; e<epochs; e++) {
            
            const batchGenerator = getMiniBatch(X_train, Y_train, batch_size)
            let batch_idx = 0;
            
            for (const batch of batchGenerator) {
                while(getIsPaused()) {
                    if(getStopTraining()) {
                        console.log("training stopped during pause")
                        return;
                    }
                    await new Promise(resolve => setTimeout(resolve, 200));
                }
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

                    let hasInvalidGrad = false;
                    for (const gradVal of p.grad) {
                        if (isNaN(gradVal) || !isFinite(gradVal)) {
                            console.warn("Invalid gradient found. halting update for this batch", p);
                            hasInvalidGrad = true;
                            break;
                        }
                    }
                    if(hasInvalidGrad) continue;


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

                if(callbacks?.onBatchEnd) {
                    let accuracy = multiClass ? calcMultiClassAccuracy(Y_pred, Y_batch) : calcBinaryAccuracy(Y_pred, Y_batch);

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

                    callbacks.onBatchEnd({
                        epoch: e,
                        batch_idx: batch_idx,
                        loss: loss.data[0],
                        accuracy: accuracy,
                        iterTime: iterTime,
                        visData: {
                            sampleX: sampleX,
                            sampleY_label: sampleY_label,
                            layerOutputs: model.getLayerOutputs(sampleX)
                        }
                    });
                }
                batch_idx++;
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