import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, prepare_dataset } from "../utils/utils_data.js";
import { getLayerColor, LayerType, NNLayer } from "../utils/utils_vis.js";
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
    private fc_config: HTMLElement;
    private conv_config: HTMLElement;
    private common_config: HTMLCollectionOf<Element>;
    private config: HTMLElement;
    private placeholder: HTMLElement;

    constructor() {
        this.container = document.getElementById('network-container')!;
        this.config_panel = document.getElementById('config-panel')!;
        this.config = document.getElementById('config')!;
        this.fc_config = document.getElementById('fc-config')!;
        this.conv_config = document.getElementById('conv-config')!;
        this.common_config = document.getElementsByClassName('common-config')!;
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
                break;
            case 'output':
                textContent += `${layerData.neurons} neurons\n`;
                break;
            case 'conv':
                textContent += `${layerData.out_channels} filters\n`;
                textContent += `${layerData.kernel_size}x${layerData.kernel_size} K\n`;
                textContent += `S:${layerData.stride} P:${layerData.padding}\n`;
                break;
        }
        textContent += `${layerData.activation}`
        layer_element.textContent = textContent.trim();
        return layer_element
    }

    private setupDocument() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges())
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer())
    }

    addLayer(type: LayerType) {
        if(this.layers[this.layers.length-1]?.type === 'output') {
            alert("Cannot add layers after the output layer");
            return;
        }

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
            case 'output':
                newLayer = {
                    id: id, 
                    type: type, 
                    neurons: 1, 
                    activation: 'sigmoid', 
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
    }

    private deselectLayer() {
        this.selected_layer = null;
        this.layers.forEach(l=>l.element.classList.remove('selected'));

        this.placeholder.style.display = 'block';
        this.config.querySelectorAll('.layer-info, .form-group, button').forEach(el=> {
            (el as HTMLElement).style.display = 'none';
        })
    }

    private updateConfigPanel(layer: NNLayer) {
        const activation_select = document.getElementById('activation-select') as HTMLSelectElement;
        const layer_position = document.getElementById('layer-position') as HTMLElement;
        const layer_type = document.getElementById('layer-type') as HTMLElement;

        // Update layer info
        const layer_index = this.layers.findIndex(l => l.id === layer.id);
        layer_position.textContent = `${layer_index + 1}`
        layer_type.textContent = layer.type;
        activation_select.value = layer.activation;

        for (let i=0; i<this.common_config.length; i++) {
            (this.common_config[i] as HTMLElement).style.display = 'block';
        }

        switch (layer.type) {
            case 'dense':
                const neuronsDense = document.getElementById('neurons-input') as HTMLInputElement;
                neuronsDense.value = layer.neurons.toString();

                this.fc_config.querySelectorAll('.layer-info, .form-group, button').forEach(el => {
                    (el as HTMLElement).style.display = 'block';
                });

                this.conv_config.querySelectorAll('.layer-info, .form-group, button').forEach(el=> {
                    (el as HTMLElement).style.display = 'none';
                })
                break;
            case 'output':
                const neuronsOutput = document.getElementById('neurons-input') as HTMLInputElement;
                neuronsOutput.value = layer.neurons.toString();

                this.fc_config.querySelectorAll('.layer-info, .form-group, button').forEach(el => {
                    (el as HTMLElement).style.display = 'block';
                });

                this.conv_config.querySelectorAll('.layer-info, .form-group, button').forEach(el=> {
                    (el as HTMLElement).style.display = 'none';
                })
                break;
            case 'conv':
                const outChannelsInput = document.getElementById('out-channels-input') as HTMLInputElement;
                const kernelSizeInput = document.getElementById('kernel-size-input') as HTMLInputElement;
                const strideInput = document.getElementById('stride-input') as HTMLInputElement;
                const paddingInput = document.getElementById('padding-input') as HTMLInputElement;

                outChannelsInput.value = layer.out_channels.toString();
                kernelSizeInput.value = layer.kernel_size.toString();
                strideInput.value = layer.stride.toString();
                paddingInput.value = layer.padding.toString();

                this.conv_config.querySelectorAll('.layer-info, .form-group, button').forEach(el => {
                    (el as HTMLElement).style.display = 'block';
                });

                this.fc_config.querySelectorAll('.layer-info, .form-group, button').forEach(el => {
                    (el as HTMLElement).style.display = 'none';
                });
                break;
        }
        
        this.placeholder.style.display = 'none';
    }

    private applyLayerChanges() {
        if (!this.selected_layer) return;

        const activation = (document.getElementById('activation-select') as HTMLSelectElement).value;
        this.selected_layer.activation = activation;

        switch (this.selected_layer.type) {
            case 'dense':
                const neuronsDense = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
                this.selected_layer.neurons = neuronsDense;
                break;
            case 'output':
                const neuronsOutput = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
                this.selected_layer.neurons = neuronsOutput;
                break;
            case 'conv':
                const outChannelsInput = document.getElementById('out-channels-input') as HTMLInputElement;
                const kernelSizeInput = document.getElementById('kernel-size-input') as HTMLInputElement;
                const strideInput = document.getElementById('stride-input') as HTMLInputElement;
                const paddingInput = document.getElementById('padding-input') as HTMLInputElement;

                this.selected_layer.out_channels = parseInt(outChannelsInput?.value ?? '0');
                this.selected_layer.kernel_size = parseInt(kernelSizeInput?.value ?? '0');
                this.selected_layer.stride = parseInt(strideInput?.value ?? '0');
                this.selected_layer.padding = parseInt(paddingInput?.value ?? '0');
                break;
        }

        const updatedElement = this.createLayerElement(this.selected_layer);
        this.selected_layer.element.textContent = updatedElement.textContent;
    }

    validateNetwork(): string[] {
        const errors: string[] = [];
        if (this.layers.length<2)   errors.push("Need atleast 2 layers");
        if (!this.layers.some(l=>l.type === 'output')) errors.push("Missing output layer");
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

    if (getStopTraining()) {
        if (runStopButton) runStopButton.textContent = 'Run Model';
        return;
    }

    if (!(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN)) {
        console.error("Datasets not found. Please select a trainset and a testset");
        return;
    }

    if (!VISUALIZER) {
        console.error("Visulizer has not yet loaded for it to run the model.");
        return;
    }
    
    const [train_x, train_y, test_x, test_y] = prepare_dataset();
    const nin = train_x.shape[1];   // train_x.shape = [m, nin]
    const LEARNING_RATE = parseFloat((document.getElementById('learning-rate') as HTMLInputElement).value) || 0.01;
    const ITERATIONS = parseInt((document.getElementById('iterations') as HTMLInputElement).value) || 500;
    const LOG_FREQ = 10;
    const VIS_FREQUENCY = 50;

    const model = createEngineModelFromVisualizer(VISUALIZER, nin);

    startTraining();
    if (runStopButton) runStopButton.textContent = 'Stop training';
    if (statusElement) statusElement.textContent = 'Preparing...';

    try {
        await trainModel(
            model, 
            train_x, 
            train_y, 
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

    const width = canvas.width
    const height = canvas.height
    ctx.clearRect(0, 0, width, height);

    if (activationData.length === 0)    return;

    let minVal = activationData[0];
    let maxVal = activationData[0];
    for (let i=1; i<activationData.length; i++) {
        if(activationData[i] < minVal) minVal = activationData[i];
        if(activationData[i] > maxVal) maxVal = activationData[i];
    }
    
    const range = maxVal - minVal;
    const scale = range === 0? 1: 255/range;

    if (renderType === 'heatmap1d' && gridWidth) {
        const blockWidth = width / gridWidth;
        for (let i=0; i<gridWidth && i<activationData.length; i++) {
            const isRelu = minVal >= 0;
            const grayVal = isRelu 
                          ? Math.max(0, Math.min(255, Math.round(activationData[i] * scale)))
                          : Math.max(0, Math.min(255, Math.round((activationData[i] - minVal)*scale)))

            ctx.fillStyle = `rgb(${grayVal}, ${grayVal}, ${grayVal})`;
            ctx.fillRect(i*blockWidth, 0, blockWidth, height);
        }
    } else if(renderType === 'feature_map' && gridHeight && gridWidth) {
        if (activationData.length !== gridWidth * gridHeight) return;
        const imageData = ctx.createImageData(gridWidth, gridHeight);
        const data = imageData.data;
        for (let i=0; i<activationData.length; i++) {
            const isRelu = minVal >= 0;
            const grayVal = isRelu
                          ? Math.max(0, Math.min(255, Math.round(activationData[i] * scale)))
                          : Math.max(0, Math.min(255, Math.round((activationData[i] - minVal)*scale)));
            const pixelIdx = i*4;
            data[pixelIdx] = grayVal; 
            data[pixelIdx+1] = grayVal; 
            data[pixelIdx+2] = grayVal;
            data[pixelIdx + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
    }
} 


function updateTrainingStatusUI(
    iter: number,
    loss: number,
    accuracy: number,
    activationData?: VISActivationData[]
) {
    if (statusElement) {
        statusElement.textContent = `Training... Iter: ${iter}, Loss: ${loss.toFixed(4)}, Acc: ${isNaN(accuracy) ? 'N/A' : accuracy.toFixed(1)}%`;
    }

    if (activationData && VISUALIZER) {
        const vizLayers = (VISUALIZER as any).layers as NNLayer[];

        activationData.forEach(actData => {
            if (actData.layerIdx < vizLayers.length) {
                const vizLayer = vizLayers[actData.layerIdx];
                const layerElement = vizLayer.element;

                let canvas = layerElement.querySelector('.activation-canvas') as HTMLCanvasElement;
                if (!canvas) {
                    canvas = document.createElement('canvas');  
                    
                    canvas.className = 'activation-canvas';
                    canvas.style.width = '90%';
                    canvas.style.height = actData.layerType === 'dense' ? '10px' : '40px';
                    canvas.style.marginTop = '5px';
                    canvas.style.border = '1px solid #555';
                    canvas.style.display = 'block';
                    canvas.style.marginLeft = 'auto';
                    canvas.style.marginRight = 'auto';
                    layerElement.appendChild(canvas);
                }

                if (actData.activationSample) {
                    if (actData.layerType === 'dense') {
                        canvas.width = Math.min(actData.activationSample.length, 200);
                        canvas.height = 10;
                        renderToCanvas(actData.activationSample, canvas, 'heatmap1d', canvas.width);
                    } else if (actData.layerType === 'conv') {
                        if (actData.shape.length === 4 && actData.shape[0] === 1) {
                            const C = actData.shape[1];
                            const H = actData.shape[2];
                            const W = actData.shape[3];
                            const mapSize = H * W;
                            if (mapSize > 0 && actData.activationSample.length >= mapSize) {
                                const firstMapData = actData.activationSample.slice(0, mapSize);
                                canvas.width = W;
                                canvas.height = H;
                                renderToCanvas(firstMapData, canvas, 'feature_map', W, H);
                            }
                        }
                    } else {
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = '#555';
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = '#ccc';
                            ctx.font = '8px sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('N/A', canvas.width / 2, canvas.height / 2 + 3);
                        }
                    }
                } else {
                    const ctx = canvas.getContext('2d');
                    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }
        });
    }
}