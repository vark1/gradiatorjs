import { visPackage } from "../types_and_interfaces/vis_interfaces.js";
import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset, prepareMNISTData } from "../utils/utils_datasets.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel } from "../nn/train.js";
import * as numUtil from "../utils/utils_num.js";
import { setTrainingState, getIsPaused, getIsTraining } from "../nn/state_management.js";
import { renderNetworkGraph } from "./computational_graph.js";
import { Sequential } from "../nn/nn.js";
import { NeuralNetworkVisualizer } from "./neuralNetworkVisualizer.js";
import { LossGraph } from "./loss_graph.js";
import { NetworkParams } from "types_and_interfaces/general.js";

const mainActionBtn = <HTMLButtonElement>document.getElementById('main-action-btn');
const stopBtn = <HTMLButtonElement>document.getElementById('stop-btn');
const statusElement = <HTMLElement>document.getElementById('training-status');

function updateBtnStates() {
    if (!mainActionBtn || !stopBtn) return;

    if (!getIsTraining()) {
        mainActionBtn.textContent = 'Start Training';
        mainActionBtn.style.backgroundColor = 'var(--accent-green)';
        stopBtn.style.display = 'none';
    } else {
        stopBtn.style.display = 'block';
        if (getIsPaused()) {
            mainActionBtn.textContent = 'Resume';
            mainActionBtn.style.backgroundColor = 'var(--accent-blue)';
        } else {
            mainActionBtn.textContent = 'Pause';
            mainActionBtn.style.backgroundColor = 'var(--accent-yellow)';
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

async function handleTraining() {

    setTrainingState('TRAINING');
    updateBtnStates();

    // if (!(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN)) { console.error("Datasets not found. Please select a trainset and a testset"); return; }
    if (!VISUALIZER) { console.error("Visulizer has not yet loaded for it to run the model."); return; }
    if (lossGraph) lossGraph.reset();
    
    const [mnist_x_train, mnist_y_train] = await prepareMNISTData();
    // const catvnoncat_data = catvnoncat_prepareDataset();

    const X_train = mnist_x_train// catvnoncat_data['train_x_og'];
    const Y_train = mnist_y_train// catvnoncat_data['train_y'];
    const [model, multiClass] = createEngineModelFromVisualizer(VISUALIZER, X_train);
    
    let lossfn = multiClass? numUtil.crossEntropyLoss_softmax: numUtil.crossEntropyLoss_binary;
    const params: NetworkParams = {
        loss_fn: lossfn,
        l_rate: parseFloat((<HTMLInputElement>document.getElementById('learning-rate')).value) || 0.01,
        epochs: parseInt((<HTMLInputElement>document.getElementById('epoch')).value) || 500,
        batch_size: parseInt((<HTMLInputElement>document.getElementById('batch-size')).value) || 100,
        update_ui_freq: 10,
        vis_freq: 50,
        multiClass: multiClass
    }

    try {
        await trainModel(model, X_train, Y_train, params, updateTrainingStatusUI, updateActivationVis);
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