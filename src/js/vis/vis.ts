import { visPackage } from "../types_and_interfaces/vis_interfaces.js";
import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset, prepareMNISTData } from "../utils/utils_datasets.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel } from "../nn/train.js";
import { crossEntropyLoss } from "../utils/utils_num.js";
import { endTraining, getIsTraining, getStopTraining,requestStopTraining, startTraining } from "../nn/training_controller.js";
import { renderNetworkGraph } from "./computational_graph.js";
import { Sequential } from "../nn/nn.js";
import { NeuralNetworkVisualizer } from "./neuralNetworkVisualizer.js";

let VISUALIZER: NeuralNetworkVisualizer;

document.addEventListener('DOMContentLoaded', () => {
    VISUALIZER = new NeuralNetworkVisualizer()
})

const runStopButton = document.getElementById('run-model-btn') as HTMLButtonElement;
if (!runStopButton) {
    console.error("Run/Stop button not found");
    requestStopTraining();
}
runStopButton.addEventListener('click', handleRunClick);

const statusElement = document.getElementById('training-status') as HTMLElement;
if (!statusElement) {
    console.error("Training Status Element not found");
    requestStopTraining();
}

async function handleRunClick() {
    if (getIsTraining()) {
        console.log("Stop button clicked: Requesting stop")
        requestStopTraining();
        statusElement.textContent = 'Stopping...';
        return;
    }

    if (getStopTraining()) { runStopButton.textContent = 'Run Model'; return; }
    // if (!(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN)) { console.error("Datasets not found. Please select a trainset and a testset"); return; }
    if (!VISUALIZER) { console.error("Visulizer has not yet loaded for it to run the model."); return; }
    
    const LEARNING_RATE = parseFloat((document.getElementById('learning-rate') as HTMLInputElement).value) || 0.01;
    const EPOCH = parseInt((document.getElementById('epoch') as HTMLInputElement).value) || 500;
    const UPDATEUI_FREQ = 10;
    const VIS_FREQ = 50;
    const BATCH_SIZE = parseInt((document.getElementById('batch-size') as HTMLInputElement).value) || 100;
    
    const [mnist_x_train, mnist_y_train] = await prepareMNISTData();
    // const catvnoncat_data = catvnoncat_prepareDataset();

    const X_train = mnist_x_train// catvnoncat_data['train_x_og'];
    const Y_train = mnist_y_train// catvnoncat_data['train_y'];
    const model = createEngineModelFromVisualizer(VISUALIZER, X_train);

    startTraining();
    runStopButton.textContent = 'Stop training';
    statusElement.textContent = 'Preparing...';
    try {
        await trainModel(
            model, 
            X_train, 
            Y_train, 
            crossEntropyLoss, 
            LEARNING_RATE, 
            EPOCH,
            BATCH_SIZE, 
            UPDATEUI_FREQ,
            VIS_FREQ,
            updateTrainingStatusUI,
            updateActivationVis
        )

        if (statusElement && !getStopTraining()) {
            statusElement.textContent = 'Training finished.'
        }
    } catch (error: any) {
        console.error("Training failed:", error);
        statusElement.textContent = `Error: ${error.message || error}`
        if (getIsTraining()) { endTraining(); }
        throw error;
    } finally {
        endTraining();
        if (runStopButton) runStopButton.textContent = 'Run Model';
    }
    console.log(model);
}

const activationPanelContainer = document.getElementById('activation-details-panel');

function updateTrainingStatusUI(epoch: number, batch_idx: number, loss: number, accuracy: number, iterTime: number) {
    statusElement.textContent = `
    Epoch ${epoch + 1}, \n
    Batch ${batch_idx}: \n
    Loss=${loss.toFixed(4)}, 
    Acc=${accuracy.toFixed(2)}%, 
    Time per batch=${(iterTime/1000).toFixed(4)}s`
}

function updateActivationVis(model: Sequential, visData: visPackage) {
    if (!VISUALIZER)                { console.error('Visualizer not found'); return; }
    if (!activationPanelContainer)  { console.error('Activation panel container not found'); return; }

    const graphContainer = document.getElementById('graph-container')
    if (!graphContainer) return;

    const { sampleX, sampleY_label, layerOutputs } = visData;
    renderNetworkGraph(graphContainer, layerOutputs, model, sampleX, sampleY_label)
}