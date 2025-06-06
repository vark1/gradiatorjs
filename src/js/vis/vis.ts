import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset } from "../utils/utils_datasets.js";
import { drawActivations, getLayerColor, LayerType, NNLayer } from "../utils/utils_vis.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel, VISActivationData } from "../nn/train.js";
import { crossEntropyLoss } from "../utils/utils_num.js";
import { endTraining, getIsTraining, getStopTraining,requestStopTraining, startTraining } from "../nn/training_controller.js";

let VISUALIZER: NeuralNetworkVisualizer;

export class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private selected_layer: NNLayer | null = null;
    private config_panel: HTMLElement;
    private placeholder: HTMLElement;

    constructor() {
        this.container = document.getElementById('network-container')!;
        this.config_panel = document.getElementById('config-panel')!;
        this.placeholder = document.getElementById('placeholder')!;

        document.addEventListener('click', (e)=> {
            // To ignore clicks inside the configuration panel
            if (this.config_panel.contains(e.target as Node)) {
                return;
            }

            const layer_element = (e.target as HTMLElement).closest('.layer') as HTMLElement | null;
            if(layer_element) {
                this.selectLayer(layer_element.dataset.id!);
            } else {
                this.deselectLayer();
            }
        })

        this.setupDocument();
    }

    private createLayerElement(layerData: NNLayer): HTMLElement {
        const layer_element = document.createElement('div');
        layer_element.className = `layer ${layerData.type}`
        layer_element.style.background = getLayerColor(layerData.type)
        layer_element.dataset.id = layerData.id;
        
        let textContent = `${layerData.type}\n`;
        switch(layerData.type) {
            case 'dense':
                textContent += `${layerData.neurons} neurons\n`;
                textContent += `${layerData.activation}`
                break;
            case 'conv':
                textContent += `Filters:${layerData.out_channels}\n`;
                textContent += `K:${layerData.kernel_size}x${layerData.kernel_size}\n`;
                textContent += `S:${layerData.stride}\nP:${layerData.padding}\n`;
                textContent += `${layerData.activation}`
                break;
            case 'flatten':
                break;
            case 'maxpool':
                textContent += `Pool size:${layerData.pool_size}\n`;
                textContent += `S:${layerData.stride}\n`;
                break;
        }
        layer_element.textContent = textContent.trim();
        return layer_element;
    }

    private setupDocument() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-flatten')?.addEventListener('click', ()=> this.addLayer('flatten'));
        document.getElementById('add-maxpool')?.addEventListener('click', ()=> this.addLayer('maxpool'));
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges())
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer())
    }

    addLayer(type: LayerType) {
        let newLayer: NNLayer;
        const id = `layer-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
        switch(type) {
            case 'dense':
                newLayer = {
                    id: id, 
                    type: type, 
                    neurons: 8, 
                    activation: 'relu', 
                    element: null as any
                };
                break;
            case 'conv':
                const outChannels = 16;
                newLayer = {
                    id: id, 
                    type: type, 
                    out_channels: outChannels, 
                    kernel_size: 3, 
                    stride: 1, 
                    padding: 0, 
                    activation: 'relu', 
                    element: null as any
                };
                break;
            case 'flatten':
                newLayer = {
                    id: id,
                    type: type,
                    element: null as any
                }
                break;
            case "maxpool":
                newLayer = {
                    id: id,
                    type: type,
                    element: null as any,
                    pool_size: 2,
                    stride: 1
                }
                break;
            default:
                console.error("Unknown layer type:", type);
                return;
        }

        const layer_element = this.createLayerElement(newLayer)
        newLayer.element = layer_element;

        this.layers.push(newLayer)
        this.container.appendChild(layer_element);
    }

    deleteSelectedLayer() {
        if (!this.selected_layer) return;

        if(confirm('Are you sure you want to delete this layer?')) {
            this.removeLayer(this.selected_layer.id);
            this.deselectLayer();
        }
    }

    private removeLayer(layer_id: string) {
        const layer_index = this.layers.findIndex(layer => layer.id === layer_id);
        if (layer_index === -1) return; //layer not found

        const layer_element = this.layers[layer_index].element;
        this.container.removeChild(layer_element);

        this.layers.splice(layer_index, 1);
    }

    private selectLayer(layer_id: string) {
        const layer = this.layers.find(l=>l.id===layer_id);
        if (!layer) return;

        this.deselectLayer();   // to remove any previous instance of selected layer

        this.selected_layer = layer;

        this.layers.forEach(l=>l.element.classList.remove('selected'));
        layer.element.classList.add('selected');
        this.updateConfigPanel(layer);
        this.showLayerConfig(layer.type);
    }

    private deselectLayer() {
        this.hideAllConfigPanels();
        this.selected_layer = null;
        this.layers.forEach(l=>l.element.classList.remove('selected'));
        this.placeholder.style.display = 'block';
    }

    private hideAllConfigPanels() {
        const layerConfigs = document.querySelectorAll('.layer-config');
        layerConfigs.forEach(p => {
            p.classList.remove('show');
        });
    }

    private showLayerConfig(layer: LayerType) {

        this.hideAllConfigPanels();

        const allLayerConfigs = document.getElementsByClassName('all');
        const layerSpecificConfigs = document.getElementsByClassName(layer);

        for (let i=0; i<allLayerConfigs.length; i++) {
            allLayerConfigs[i].classList.add('show');
        }
        for (let i=0; i<layerSpecificConfigs.length; i++) {
            layerSpecificConfigs[i].classList.add('show');
        }
    }

    private updateConfigPanel(layer: NNLayer) {
        const layer_position = document.getElementById('layer-position') as HTMLElement;
        const layer_type = document.getElementById('layer-type') as HTMLElement;
        const activation_select = document.getElementById('activation-select') as HTMLSelectElement;

        // Update layer info
        const layer_index = this.layers.findIndex(l => l.id === layer.id);
        layer_position.textContent = `${layer_index + 1}`
        layer_type.textContent = layer.type;

        switch (layer.type) {
            case 'dense':
                const neuronsDense = document.getElementById('neurons-input') as HTMLInputElement;
                neuronsDense.value = layer.neurons.toString();
                activation_select.value = layer.activation;
                break;
            case 'conv':
                const outChannelsInput = document.getElementById('out-channels-input') as HTMLInputElement;
                const kernelSizeInput = document.getElementById('kernel-size-input') as HTMLInputElement;
                const strideConvInput = document.getElementById('stride-conv-input') as HTMLInputElement;
                const paddingInput = document.getElementById('padding-input') as HTMLInputElement;
                
                activation_select.value = layer.activation;
                outChannelsInput.value = layer.out_channels.toString();
                kernelSizeInput.value = layer.kernel_size.toString();
                strideConvInput.value = layer.stride.toString();
                paddingInput.value = layer.padding.toString();
                break;
            case 'maxpool':
                const poolSizeInput = document.getElementById('pool-size-input') as HTMLInputElement;
                const strideMaxpoolInput = document.getElementById('stride-maxpool-input') as HTMLInputElement;

                poolSizeInput.value = layer.pool_size.toString();
                strideMaxpoolInput.value = layer.stride.toString();
                break;
            case 'flatten':
                break;
        }
        
        this.placeholder.style.display = 'none';
    }

    private applyLayerChanges() {
        if (!this.selected_layer) return;

        const activation = (document.getElementById('activation-select') as HTMLSelectElement).value;

        switch (this.selected_layer.type) {
            case 'dense':
                const neuronsDense = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
                this.selected_layer.neurons = neuronsDense;
                this.selected_layer.activation = activation;
                break;
            case 'conv':
                const outChannelsInput = document.getElementById('out-channels-input') as HTMLInputElement;
                const kernelSizeInput = document.getElementById('kernel-size-input') as HTMLInputElement;
                const strideConvInput = document.getElementById('stride-conv-input') as HTMLInputElement;
                const paddingInput = document.getElementById('padding-input') as HTMLInputElement;

                this.selected_layer.out_channels = parseInt(outChannelsInput?.value ?? '0');
                this.selected_layer.kernel_size = parseInt(kernelSizeInput?.value ?? '0');
                this.selected_layer.stride = parseInt(strideConvInput?.value ?? '0');
                this.selected_layer.padding = parseInt(paddingInput?.value ?? '0');
                this.selected_layer.activation = activation;
                break;

            case 'maxpool':
                const poolSizeInput = document.getElementById('pool-size-input') as HTMLInputElement;
                const strideMaxpoolInput = document.getElementById('stride-maxpool-input') as HTMLInputElement;

                this.selected_layer.pool_size = parseInt(poolSizeInput?.value ?? '0');
                this.selected_layer.stride = parseInt(strideMaxpoolInput?.value ?? '0');
                break;
        }

        const updatedElement = this.createLayerElement(this.selected_layer);
        this.selected_layer.element.textContent = updatedElement.textContent;
    }

    validateNetwork(): string[] {
        const errors: string[] = [];
        if (this.layers.length<2)   errors.push("Need atleast 2 layers");
        return errors;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    VISUALIZER = new NeuralNetworkVisualizer()
})

const runStopButton = document.getElementById('run-model-btn') as HTMLButtonElement;
if (runStopButton) {
    runStopButton.addEventListener('click', handleRunClick);
} else {
    console.error("Run/Stop button not found");
}

const statusElement = document.getElementById('training-status');

async function handleRunClick() {
    if (getIsTraining()) {
        console.log("Stop button clicked: Requesting stop")
        requestStopTraining();
        if (statusElement) statusElement.textContent = 'Stopping...';
        return;
    }

    if (getStopTraining()) { if (runStopButton) runStopButton.textContent = 'Run Model'; return; }
    if (!(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN)) { console.error("Datasets not found. Please select a trainset and a testset"); return; }
    if (!VISUALIZER) { console.error("Visulizer has not yet loaded for it to run the model."); return; }
    
    const X = catvnoncat_prepareDataset();
    const LEARNING_RATE = parseFloat((document.getElementById('learning-rate') as HTMLInputElement).value) || 0.01;
    const ITERATIONS = parseInt((document.getElementById('iterations') as HTMLInputElement).value) || 500;
    const LOG_FREQ = 10;
    const VIS_FREQUENCY = 50;

    const model = createEngineModelFromVisualizer(VISUALIZER, X['train_x_og']);

    startTraining();
    if (runStopButton) runStopButton.textContent = 'Stop training';
    if (statusElement) statusElement.textContent = 'Preparing...';

    try {
        await trainModel(
            model, 
            X['train_x_og'], 
            X['train_y'], 
            crossEntropyLoss, 
            LEARNING_RATE, 
            ITERATIONS, 
            LOG_FREQ,
            VIS_FREQUENCY,
            updateTrainingStatusUI
        )

        if (statusElement && !getStopTraining()) {
            statusElement.textContent = 'Training finished.'
        }
    } catch (error: any) {
        console.error("Training failed:", error);
        if (statusElement) statusElement.textContent = `Error: ${error.message || error}`
        if (getIsTraining()) { endTraining(); }
        throw error;
    } finally {
        endTraining();
        if (runStopButton) runStopButton.textContent = 'Run Model';
    }

    console.log(model);
}

function renderToCanvas(
    activationData: Float64Array,
    canvas: HTMLCanvasElement,
    renderType: 'heatmap1d' | 'feature_map',
    gridWidth?: number,
    gridHeight?: number
): void {
    const ctx = canvas.getContext('2d')
    if (!ctx) return;

    const canvasWidth = canvas.width
    const canvasHeight = canvas.height
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!activationData || activationData.length === 0) {
        ctx.fillStyle = '#ddd'; 
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        return;
    }

    let minVal = activationData[0];
    let maxVal = activationData[0];
    for (let i=1; i<activationData.length; i++) {
        if(activationData[i] < minVal) minVal = activationData[i];
        if(activationData[i] > maxVal) maxVal = activationData[i];
    }
    
    const range = maxVal - minVal;
    const scale = range === 0? 1: 255/range;

    if (renderType === 'heatmap1d' && gridWidth) {
        const blockWidth = canvasWidth / gridWidth;
        for (let i=0; i<gridWidth && i<activationData.length; i++) {
            const normalizedVal = range === 0 ? 0.5 : (activationData[i] - minVal)/range;
            const grayVal = Math.max(0, Math.min(255, Math.round(normalizedVal * 255)))
            ctx.fillStyle = `rgb(${grayVal}, ${grayVal}, ${grayVal})`;
            ctx.fillRect(Math.floor(i*blockWidth), 0, Math.ceil(blockWidth), canvasHeight);
        }
    } else if(renderType === 'feature_map' && gridHeight && gridWidth) {
        if (activationData.length !== gridWidth * gridHeight) {
            console.warn(`Feature map data size mismatch for rendering: ${activationData.length} vs ${gridWidth * gridHeight}`)
            ctx.fillStyle = '#fdd'; ctx.fillRect(0,0, canvasWidth, canvasHeight); 
            return;
        }

        const imageData = ctx.createImageData(gridWidth, gridHeight);
        const data = imageData.data;
        for (let i=0; i<activationData.length; i++) {
            const normalizedValue = range === 0 ? 0.5 : (activationData[i] - minVal) / range;
            const grayVal = Math.max(0, Math.min(255, Math.round(normalizedValue * 255)));
            const pixelIdx = i*4;
            data[pixelIdx] = grayVal; 
            data[pixelIdx+1] = grayVal; 
            data[pixelIdx+2] = grayVal;
            data[pixelIdx + 3] = 255;
        }

        ctx.imageSmoothingEnabled = false;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = gridWidth;
        tempCanvas.height = gridHeight;
        tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
        ctx.drawImage(tempCanvas, 0, 0, canvasWidth, canvasHeight);
    } else {
        ctx.fillStyle = '#eee'; ctx.fillRect(0,0, canvasWidth, canvasHeight);
        ctx.fillStyle = 'red'; ctx.fillText('Render type error', 5, 10);
    }
}

// helper to render fallback text on canvas
function renderFallbackText(canvas: HTMLCanvasElement, text: string) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#e0e0e0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#555';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width/2, canvas.height/2);
    }
}

const activationPanelContainer = document.getElementById('activation-details-panel');

function updateTrainingStatusUI(
    iter: number,
    loss: number,
    accuracy: number,
    activationData?: VISActivationData[]
) {
    if (statusElement) {
        statusElement.textContent = `Training... Iter: ${iter}, Loss: ${loss.toFixed(4)}, Acc: ${isNaN(accuracy) ? 'N/A' : accuracy.toFixed(1)}%`;
    }

    if (activationData && VISUALIZER && activationPanelContainer) {
        const vizLayers = (VISUALIZER as any).layers as NNLayer[];

        activationData.forEach(actData => {
            if (actData.layerIdx < vizLayers.length) {
                const vizLayer = vizLayers[actData.layerIdx];
                const layerElement = vizLayer.element;

                let layerActivationContainerId = `act-vis-layer-${vizLayer.id}`;
                let layerActivationContainer = document.getElementById(layerActivationContainerId);

                if (!layerActivationContainer) {
                    layerActivationContainer = document.createElement('div');
                    layerActivationContainer.id = layerActivationContainerId;

                    const title = document.createElement('h4');
                    title.textContent = `Layer ${actData.layerIdx + 1}: ${vizLayer.type}`;
                    layerActivationContainer.appendChild(title);

                    const canvasWrapper = document.createElement('div');
                    canvasWrapper.className = 'activation-maps-wrapper';
                    layerActivationContainer.appendChild(canvasWrapper);

                    activationPanelContainer.appendChild(layerActivationContainer);
                }

                const canvasWrapper = layerActivationContainer.querySelector('.activation-maps-wrapper') as HTMLElement;
                if (!canvasWrapper) return;

                let canvasId = `act-canvas-${vizLayer.id}-map0`; // ID for the first map/heatmap
                let canvas = document.getElementById(canvasId) as HTMLCanvasElement;
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.id = canvasId;
                    canvas.style.border = '1px solid #ccc';
                    canvas.style.backgroundColor = '#f8f8f8';
                    canvasWrapper.appendChild(canvas);
                }

                if (!actData.activationSample || actData.activationSample.length === 0) {
                    renderFallbackText(canvas, 'no activation data');
                    return;
                }

                if (actData.layerType === 'dense') {
                    const numActivations = actData.activationSample.length;
                    canvas.style.width = '100%';
                    canvas.style.height = '15px';
                    canvas.width = Math.min(numActivations, 256);
                    canvas.height = 15;
                    renderToCanvas(actData.activationSample, canvas, 'heatmap1d', numActivations);
                } else if (actData.layerType === 'conv') {
                    if (actData.shape.length === 4 && actData.shape[0] === 1) {

                        drawActivations(activationPanelContainer, actData, 2)
                        
                    } else {
                        renderFallbackText(canvas, `Conv shape error (${actData.shape.join(',')})`);
                    }
                } else {
                    renderFallbackText(canvas, `Type? (${actData.layerType})`);
                }
            }
        });
    }
}