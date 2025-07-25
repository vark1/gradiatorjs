import { Val } from "./val.js";
import { Sequential } from "./model.js";
import { getStopTraining, endTraining, getIsPaused, getTrainingContext, advanceEpoch } from "./state_management.js";
import { calcBinaryAccuracy, calcMultiClassAccuracy } from "./accuracy.js";
import type { NetworkParams, Messenger } from "./types.js";
import { assert } from "./utils.js";

function createBatchVal(ogVal: Val, batchIndices: number[], currentBatchSize: number, features: number) {
    const batchShape = [currentBatchSize, ...ogVal.shape.slice(1)];
    const batchVal = new Val(batchShape);

    for (let k=0; k<currentBatchSize; k++) {
        const ogIdx = batchIndices[k]!;
        const sourceOffset = ogIdx*features;
        const destOffset = k*features;
        batchVal.data.set(ogVal.data.subarray(sourceOffset, sourceOffset+features), destOffset);
    }
    return batchVal;
}

// generator that will yield mini-batches from the full dataset 
export function* getMiniBatch(X: Val, Y: Val, batchSize: number, shuffle: boolean = true){
    const numSamples = X.shape[0]!;
    const indices = Array.from({ length: numSamples }, (_, i) => i);

    if (shuffle) {
        for (let i=indices.length-1; i>0; i--) {
            const j=Math.floor(Math.random()*(i+1));
            [indices[i], indices[j]] = [indices[j]!, indices[i]!];
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

function getActSample(batch: {x: Val, y: Val}): [Val, number] {
    const sampleX = new Val(batch.x.shape.slice(1)).reshape([1, ...batch.x.shape.slice(1)]);
    sampleX.data = batch.x.data.slice(0, batch.x.size / batch.x.shape[0]!);

    let sampleY_label = -1;
    const y_features = batch.y.size / batch.y.shape[0]!;
    if (y_features === 1) { // If label is a single number
        sampleY_label = batch.y.data[0];
    } else { // If label is one-hot encoded
        const y_sample_data = batch.y.data.slice(0, y_features);
        sampleY_label = y_sample_data.indexOf(1);
    }
    return [sampleX, sampleY_label];
}

export async function trainModel(
    model: Sequential, 
    X_train: Val, 
    Y_train: Val, 
    params: NetworkParams,
    messenger?: Messenger,
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
                            p.data[j] -= l_rate * p.grad[j]!;
                        }
                    }
                }

                const iterEndTime = performance.now();
                const iterTime = iterEndTime - iterStartTime;
                totalProcessingTime += iterTime;
                iteration++;
                if (messenger) {
                    assert(model instanceof Sequential, ()=>`Model is not an instance of sequential.`)
                    let accuracy = multiClass ? calcMultiClassAccuracy(Y_pred, Y_batch) : calcBinaryAccuracy(Y_pred, Y_batch);

                    const [sampleX, sampleY_label] = getActSample(batch);
                    
                    const rawLayerOutputs = model.getLayerOutputs(sampleX).map(val => ({
                        Zdata: val['Z']?.data.buffer,
                        Zshape: val['Z']?.shape,
                        Adata: val['A']?.data.buffer,
                        Ashape: val['A']?.shape
                    }));

                    const transferableBuffersSet = new Set<ArrayBuffer>();
                    rawLayerOutputs.forEach(layer => {
                        if (layer.Zdata) transferableBuffersSet.add(layer.Zdata);
                        if (layer.Adata) transferableBuffersSet.add(layer.Adata);
                    });
                    transferableBuffersSet.add(sampleX.data.buffer);

                    const transferableBuffers = Array.from(transferableBuffersSet);

                    messenger.postMessage({
                        type: 'batchEnd',
                        epoch: e,
                        batch_idx: iteration,
                        loss: loss.data[0],
                        accuracy: accuracy,
                        iterTime: iterTime,
                        visData: {
                            sampleX: {
                                data: sampleX.data.buffer,
                                shape: sampleX.shape
                            },
                            sampleY_label: sampleY_label,
                            layerOutputs: rawLayerOutputs
                        }
                    }, transferableBuffers);
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

export async function trainSingleBatch(messenger: Messenger): Promise<void> {
    const {model, X_train, Y_train, params, currentEpoch, batchGenerator, iteration} = getTrainingContext();

    if (getStopTraining()) {
        endTraining();
        messenger.postMessage({type: 'complete', reason: 'stopped by user'});
        return;
    }
    if (getIsPaused() || !model || !batchGenerator || !params) {
        return;
    }

    const iterStartTime = performance.now();
    const batchResult = batchGenerator.next();

    if (batchResult.done) {
        if(advanceEpoch()) {
            messenger.postMessage({type: 'epochEnd', epoch: currentEpoch});
        } else {
            messenger.postMessage({type: 'complete', reason: 'All epochs finished'})
        }
        return;
    }
    const batch = batchResult.value;
    const { x: X_batch, y: Y_batch } = batch;
    const {loss_fn, l_rate, epochs, batch_size, multiClass} = params
     

    model.zeroGrad();
    const Y_pred = model.forward(X_batch);
    const loss = loss_fn(Y_pred, Y_batch);
    loss.backward();

    const modelParams = model.parameters();
    
    for (const p of modelParams) {
        if (!p.grad || p.data.length !== p.grad.length) {
            console.warn(`Skipping update for parameter due to missing/mismatched gradient. Param shape: ${p.shape}`);
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
                p.data[j] -= l_rate * p.grad[j]!;
            }
        }
    }

    const iterEndTime = performance.now();
    const iterTime = iterEndTime - iterStartTime;

    assert(model instanceof Sequential, ()=>`Model is not an instance of sequential.`)
    let accuracy = multiClass ? calcMultiClassAccuracy(Y_pred, Y_batch) : calcBinaryAccuracy(Y_pred, Y_batch);

    const [sampleX, sampleY_label] = getActSample(batch);
    
    const rawLayerOutputs = model.getLayerOutputs(sampleX).map(val => ({
        Zdata: val['Z']?.data.buffer,
        Zshape: val['Z']?.shape,
        Adata: val['A']?.data.buffer,
        Ashape: val['A']?.shape
    }));

    const transferableBuffersSet = new Set<ArrayBuffer>();
    rawLayerOutputs.forEach(layer => {
        if (layer.Zdata) transferableBuffersSet.add(layer.Zdata);
        if (layer.Adata) transferableBuffersSet.add(layer.Adata);
    });
    transferableBuffersSet.add(sampleX.data.buffer);
    const transferableBuffers = Array.from(transferableBuffersSet);

    messenger.postMessage({
        type: 'batchEnd',
        epoch: currentEpoch,
        batch_idx: iteration,
        loss: loss.data[0],
        accuracy: accuracy,
        iterTime: iterTime,
        visData: {
            sampleX: {
                data: sampleX.data.buffer,
                shape: sampleX.shape
            },
            sampleY_label: sampleY_label,
            layerOutputs: rawLayerOutputs
        }
    }, transferableBuffers);
}