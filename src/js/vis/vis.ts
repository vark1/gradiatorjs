import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset, prepareMNISTData } from "../utils/utils_datasets.js";
import { drawActivations, getLayerColor } from "../utils/utils_vis.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel } from "../nn/train.js";
import { crossEntropyLoss } from "../utils/utils_num.js";
import { endTraining, getIsTraining, getStopTraining,requestStopTraining, startTraining } from "../nn/training_controller.js";
import { ActivationType, LayerType, NNLayer } from "../types_and_interfaces/general.js";
import { VISActivationData, SerializableNNLayer, LayerCreationOptions } from "types_and_interfaces/vis_interfaces.js";

let VISUALIZER: NeuralNetworkVisualizer;

export class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private selected_layer: NNLayer | null = null;
    private config_panel: HTMLElement;
    private placeholder: HTMLElement;

    private readonly STORAGE_KEY = 'neuralNetworkVisualizerConfig';

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

        this.setupEventListeners();
        this.loadNetworkFromLocalStorage();
    }

    private setupEventListeners() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer({type: 'dense', neurons: 2, activation: 'relu'}));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer({type: 'conv', out_channels: 8, kernel_size: 5, stride: 2, padding: 2, activation: 'relu'}));
        document.getElementById('add-flatten')?.addEventListener('click', ()=> this.addLayer({type:'flatten'}));
        document.getElementById('add-maxpool')?.addEventListener('click', ()=> this.addLayer({type:'maxpool', pool_size: 2, stride: 2}));
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges());
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer());
        
        document.getElementById('save-network-btn')?.addEventListener('click', ()=> {
            this.saveNetworkToLocalStorage();
            const statusDiv = document.getElementById('persistence-status');
            if (!statusDiv) return;
            statusDiv.textContent = 'Network saved to browser storage.';
            setTimeout(()=> { statusDiv.textContent = '';}, 2000);
        })

        document.getElementById('load-network-btn')?.addEventListener('click', ()=> {
            this.loadNetworkFromLocalStorage();
        });
        
        document.getElementById('clear-network-btn')?.addEventListener('click', ()=> {
            localStorage.removeItem(this.STORAGE_KEY);
            const statusDiv = document.getElementById('persistence-status');
            if (statusDiv) statusDiv.textContent = 'Cleared saved network from browser storage.';
            console.log("Cleared saved network.");
        })
    }

    // Converts the current network layer structure into a serializable JSON object.
    private getNetworkConfig(): SerializableNNLayer[] {
        return this.layers.map(layer => {
            const { element, ...serializableLayer} = layer;
            return serializableLayer;
        })
    }

    private loadNetworkFromLocalStorage(): void {
        const statusDiv = document.getElementById('persistence-status');
        try {
            const jsonString = localStorage.getItem(this.STORAGE_KEY);
            if (!jsonString) {
                console.log("No saved network found in localStorage.");
                if (statusDiv) statusDiv.textContent = 'No saved network found.';
                return;
            }

            const savedConfig: SerializableNNLayer[] = JSON.parse(jsonString);
            if (!Array.isArray(savedConfig)) {
                throw new Error("Saved data is not a valid network configuration.");
            }

            this.layers = []
            savedConfig.forEach(layerOptions=> {
                this.addLayer(<LayerCreationOptions>layerOptions)
            });

            console.log("successfully loaded network from localstorage")
            if (statusDiv) statusDiv.textContent = 'Loaded network from browser storage.';
        } catch (e) {
            console.error("Error loading network from localStorage:", e);
            if (statusDiv) statusDiv.textContent = `Error: Could not load network.`;
        }
    }

    private saveNetworkToLocalStorage(): void {
        try {
            const networkConfig = this.getNetworkConfig();
            const jsonString = JSON.stringify(networkConfig);
            localStorage.setItem(this.STORAGE_KEY, jsonString);
            console.log("network configuration saved")
        } catch (e) {
            console.log("Error in saving to local storage: ", e);
        }
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

    addLayer(options: LayerCreationOptions): void {
        let newLayer: NNLayer;
        const id = `layer-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;

        switch(options.type) {
            case 'dense':
                newLayer = {
                    id,
                    type: options.type, 
                    neurons: options.neurons, 
                    activation: options.activation, 
                    element: null as any
                };
                break;
            case 'conv':
                newLayer = {
                    id, 
                    type: options.type, 
                    out_channels: options.out_channels, 
                    kernel_size: options.kernel_size, 
                    stride: options.stride, 
                    padding: options.padding, 
                    activation: options.activation, 
                    element: null as any
                };
                break;
            case 'flatten':
                newLayer = {
                    id,
                    type: options.type,
                    element: null as any
                }
                break;
            case "maxpool":
                newLayer = {
                    id,
                    type: options.type,
                    pool_size: options.pool_size,
                    stride: options.stride, 
                    element: null as any,
                }
                break;
            default:
                console.error("Unknown layer type for options:", options);
                return;
        }

        const layer_element = this.createLayerElement(newLayer)
        newLayer.element = layer_element;

        this.layers.push(newLayer)
        this.container.appendChild(layer_element);
        this.saveNetworkToLocalStorage();
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

        const activation = <ActivationType>(document.getElementById('activation-select') as HTMLSelectElement).value;

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

function updateTrainingStatusUI(epoch: number, batch_idx: number, loss: number, accuracy: number, iterTime: number) {
    statusElement.textContent = `
    Epoch ${epoch + 1}, \n
    Batch ${batch_idx}: \n
    Loss=${loss.toFixed(4)}, 
    Acc=${accuracy.toFixed(2)}%, 
    Time per example=${(iterTime/1000).toFixed(4)}s`
}

function updateActivationVis(activationData: VISActivationData[]) {

    if (!activationData)            { console.error('activation data not found'); return; }
    if (!VISUALIZER)                { console.error('Visualizer not found'); return; }
    if (!activationPanelContainer)  { console.error('Activation panel container not found'); return; }

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

                    drawActivations(activationPanelContainer, actData, 4)
                    
                    const H_out = actData.shape[1];
                    const W_out = actData.shape[2];
                    const C_out = actData.shape[3];
                    const singleMapSize = H_out * W_out;

                    if (singleMapSize > 0 && C_out > 0 && actData.activationSample.length === singleMapSize * C_out) {
                        // xtracting data for the first feature map (channel 0)
                        const firstMapData = new Float64Array(singleMapSize);
                        for (let i=0; i<singleMapSize; i++) {
                            firstMapData[i] = actData.activationSample[i * C_out + 0];
                        }

                        canvas.width = W_out;
                        canvas.height = H_out;
                        canvas.style.imageRendering = 'pixelated';

                        const maxDisplayDim = 48;
                        let displayW, displayH;
                        if (W_out >= H_out) { // wider or square
                            displayW = maxDisplayDim;
                            displayH = Math.round(maxDisplayDim * (H_out / W_out));
                        } else {              // taller
                            displayH = maxDisplayDim;
                            displayW = Math.round(maxDisplayDim * (W_out / H_out));
                        }
                        canvas.style.width = `${displayW}px`;
                        canvas.style.height = `${displayH}px`;

                        renderToCanvas(firstMapData, canvas, 'feature_map', W_out, H_out);
                    } else {
                        renderFallbackText(canvas, `Conv data error (S:${actData.activationSample.length} vs E:${singleMapSize * C_out})`)
                    }
                    
                } else {
                    renderFallbackText(canvas, `Conv shape error (${actData.shape.join(',')})`);
                }
            } else {
                renderFallbackText(canvas, `Type? (${actData.layerType})`);
            }
        }
    });
}