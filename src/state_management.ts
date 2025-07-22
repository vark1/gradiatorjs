import type { Sequential } from "./model.js";
import { getMiniBatch } from "./train.js";
import type { NetworkParams } from "./types.js";
import type { Val } from "./val.js";

let isTraining = false;
let stopTraining = false;
let isPaused = false;

const trainingContext = {
    model: null as Sequential | null,
    X_train: null as Val | null,
    Y_train: null as Val | null,
    params: null as NetworkParams | null,
    currentEpoch: 0,
    batchGenerator: null as Generator<any, void, unknown> | null,
    iteration: 0
};

export type trainingState = 'IDLE' | 'TRAINING' | 'PAUSED' | 'STOPPING';

export const getIsTraining = (): boolean => isTraining;
export const getIsPaused = (): boolean => isPaused;
export const getStopTraining = (): boolean => stopTraining;

export function getTrainingContext() {
    return trainingContext;
}

export function startTraining(): void {
    isTraining = true;
    stopTraining = false;
    isPaused = false;
    console.log("training started");
}

export function requestStopTraining(): void {
    if (!isTraining) {
        console.log("training stop requested when it wasnt running in the first place")
        return;
    }
    stopTraining = true;
    isPaused = false;
    console.log("stop requested");
}

export function endTraining(): void {
    isTraining = false;
    stopTraining = false;
    isPaused = false;
    trainingContext.model = null;
    trainingContext.X_train = null;
    trainingContext.Y_train = null;
    trainingContext.params = null;
    trainingContext.currentEpoch = 0;
    trainingContext.batchGenerator = null;
    console.log("training finished");
}

export function requestPause(): void {
    if (isTraining && !isPaused) {
        isPaused = true;
        console.log("pausing training");
    }
}

export function requestResume(): void {
    if (isTraining && isPaused) {
        isPaused = false;
        console.log("resume requested")
    }
}

export function setTrainingState(newState: trainingState): void {
    switch(newState) {
        case 'IDLE':
            isTraining = false;
            stopTraining = false;
            isPaused = false;
            break;
        case 'TRAINING':
            if (!isTraining) {
                isTraining = true;
                stopTraining = false;
                isPaused = false;
            } else if (isPaused) {  // already paused, resuming
                isPaused = false;
            }
            break;
        case 'PAUSED':
            if (isTraining && !isPaused) {
                isPaused = true;
            }
            break;
        case 'STOPPING':
            if (isTraining) {
                stopTraining = true;
                isPaused = false;
            }
            break;
        }
        console.log("State changed to: ", newState);
}

export function setupTrainingContext(model: Sequential, x: Val, y: Val, params: NetworkParams): void {
    trainingContext.model = model;
    trainingContext.X_train = x;
    trainingContext.Y_train = y;
    trainingContext.params = params;
    trainingContext.currentEpoch = 0;
    trainingContext.batchGenerator = getMiniBatch(x, y, params.batch_size);
    startTraining();
    console.log("training context has been set up for Epoch 0.");
}

export function advanceEpoch(): boolean {
    if (!trainingContext.params || !trainingContext.X_train || !trainingContext.Y_train) {
        return false;
    }
    
    trainingContext.currentEpoch++;
    
    if (trainingContext.currentEpoch >= trainingContext.params.epochs) {
        console.log("Max epochs reached.");
        endTraining();
        return false;
    }
    
    // Create a new generator for the new epoch
    trainingContext.batchGenerator = getMiniBatch(
        trainingContext.X_train, 
        trainingContext.Y_train, 
        trainingContext.params.batch_size
    );
    console.log(`Advanced to Epoch ${trainingContext.currentEpoch}`);
    return true;
}