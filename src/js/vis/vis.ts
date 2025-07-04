import { TrainingProgress, visPackage } from "../types_and_interfaces/vis_interfaces.js";
import { prepareCatvnoncatData, prepareMNISTData } from "../utils/utils_datasets.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel } from "../../train.js";
import * as numUtil from "../../loss.js";
import { setTrainingState, getIsPaused, getIsTraining, getStopTraining } from "../../state_management.js";
import { renderNetworkGraph } from "./computational_graph.js";
import { Sequential } from "../../layers.js";
import { NeuralNetworkVisualizer } from "./neuralNetworkVisualizer.js";
import { LossGraph } from "./loss_graph.js";
import { NetworkParams } from "types_and_interfaces/general.js";
import { Val } from "../../val.js";

let currentModel: Sequential | null = null; 

const mainActionBtn = <HTMLButtonElement>document.getElementById('main-action-btn');
const stopBtn = <HTMLButtonElement>document.getElementById('stop-btn');
const statusElement = <HTMLElement>document.getElementById('training-status');

function downloadObjectAsJSON(exportObj: any, exportName: string): void {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
    const x = document.createElement('a');
    x.setAttribute('href', dataStr);
    x.setAttribute("download", exportName);
    document.body.appendChild(x);
    x.click();
    x.remove();
}

function updateBtnStates() {
    if (!mainActionBtn || !stopBtn) return;

    if (!getIsTraining()) {
        mainActionBtn.textContent = 'Start Training';
        stopBtn.style.display = 'none';
    } else {
        stopBtn.style.display = 'block';
        if (getIsPaused()) {
            mainActionBtn.textContent = 'Resume';
        } else {
            mainActionBtn.textContent = 'Pause';
        }
    }
}

mainActionBtn?.addEventListener('click', handleMainActionClick);
stopBtn?.addEventListener('click', handleStopClick);

function handleMainActionClick() {
    if (!getIsTraining()) {
        handleTraining();
    } else if (getIsPaused()) {
        setTrainingState('TRAINING');
    } else {
        setTrainingState('PAUSED');
        updateBtnStates();
    }
}

function handleStopClick() {
    console.log("Stop button clicked: Requesting stop");
    setTrainingState('STOPPING');
    statusElement.textContent = 'Stopping...';
    return;
}

let VISUALIZER: NeuralNetworkVisualizer;
let lossGraph: LossGraph;

document.addEventListener('DOMContentLoaded', () => {
    VISUALIZER = new NeuralNetworkVisualizer();
    lossGraph = new LossGraph('loss-accuracy-chart');
})

document.getElementById('download-model-btn')?.addEventListener('click', () => {
    if (!currentModel) {
        alert("No trained model available to downlaod. please run training first");
        return;
    }
    
    try {
        const modelJSON = currentModel.toJSON();
        downloadObjectAsJSON(modelJSON, 'trained_model.json');
    } catch (error) {
        console.error("Failed to serialize or download model:", error);
        alert("Error: Could not download model. Check console for details.");
    }
});

async function loadDataset() {
    const datasetType = (<HTMLSelectElement>document.getElementById('dataset-select')).value;
    let X = new Val([]);
    let Y = new Val([]);

    if (datasetType === 'mnist') {
        const [mnist_x_train, mnist_y_train] = await prepareMNISTData();
        X = mnist_x_train;
        Y = mnist_y_train;
    } else if (datasetType === 'catvnoncat') {
        const catvnoncat_data = await prepareCatvnoncatData();
        X = catvnoncat_data['train_x'];
        Y = catvnoncat_data['train_y'];
    }
    return [X, Y]
}

async function handleTraining() {
    if (!VISUALIZER) { console.error("Visulizer has not yet loaded for it to run the model."); return; }
    if (lossGraph) lossGraph.reset();

    setTrainingState('TRAINING');
    updateBtnStates();

    const [X, Y] = await loadDataset();

    const [model, multiClass] = createEngineModelFromVisualizer(VISUALIZER, X);
    currentModel = model;
    
    const params: NetworkParams = {
        loss_fn: multiClass? numUtil.crossEntropyLoss_softmax: numUtil.crossEntropyLoss_binary,
        l_rate: parseFloat((<HTMLInputElement>document.getElementById('learning-rate')).value) || 0.01,
        epochs: parseInt((<HTMLInputElement>document.getElementById('epoch')).value) || 500,
        batch_size: parseInt((<HTMLInputElement>document.getElementById('batch-size')).value) || 100,
        multiClass: multiClass
    }

    const trainingCallbacks = {
        onBatchEnd: (progress: TrainingProgress) => {
            updateTrainingStatusUI(progress.epoch, progress.batch_idx, progress.loss, progress.accuracy, progress.iterTime);
            if (progress.visData && currentModel) {
                updateActivationVis(currentModel, progress.visData);
            }
        }
    };

    try {
        await trainModel(currentModel, X, Y, params, trainingCallbacks);
        if (statusElement) statusElement.textContent = 'Training finished.'
    } catch (error: any) {
        console.error("Training failed:", error);
        if (statusElement) statusElement.textContent = `Error: ${error.message || error}`;
    } finally {
        setTrainingState('IDLE');
        updateBtnStates();
        console.log("handleTraining finished, state set to IDLE.");
    }
    console.log(model);
}

function updateTrainingStatusUI(epoch: number, batch_idx: number, loss: number, accuracy: number, iterTime: number) {
    statusElement.textContent = `
    Epoch ${epoch + 1}, \n
    Batch ${batch_idx}: \n
    Loss=${loss.toFixed(4)}, 
    Acc=${accuracy.toFixed(2)}%, 
    Time per batch=${(iterTime/1000).toFixed(4)}s`

    if (lossGraph) {
        lossGraph.addData(batch_idx, loss, accuracy);
    }
}

function updateActivationVis(model: Sequential, visData: visPackage) {
    if (!VISUALIZER)                { console.error('Visualizer not found'); return; }

    const graphContainer = document.getElementById('graph-container')
    if (!graphContainer) return;

    const { sampleX, sampleY_label, layerOutputs } = visData;
    renderNetworkGraph(graphContainer, layerOutputs, model, sampleX, sampleY_label)
}