let isTraining = false;
let stopTraining = false;
let isPaused = false;

export type trainingState = 'IDLE' | 'TRAINING' | 'PAUSED' | 'STOPPING';

export const getIsTraining = (): boolean => isTraining;
export const getIsPaused = (): boolean => isPaused;
export const getStopTraining = (): boolean => stopTraining;

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