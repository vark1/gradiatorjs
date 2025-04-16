let isTraining = false;
let stopTraining = false;

export function getIsTraining() : boolean {
    return isTraining;
}

export function getStopTraining() : boolean {
    return stopTraining;
}

export function startTraining() : void {
    console.log("training started");
    isTraining = true;
    stopTraining = false;
}

export function requestStopTraining() : void {
    console.log("stop requested");
    stopTraining = true;
}

export function endTraining() : void {
    console.log("training finished");
    isTraining = false;
    stopTraining = false;
}