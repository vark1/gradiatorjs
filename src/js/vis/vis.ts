import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, prepare_dataset } from "../utils/utils_data.js";
import { getLayerColor, LayerType, NNLayer } from "../utils/utils_vis.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { trainModel } from "../nn/train.js";
import { crossEntropyLoss } from "../utils/utils_num.js";
import { endTraining, getIsTraining, getStopTraining,requestStopTraining, startTraining } from "../nn/training_controller.js";

let VISUALIZER: NeuralNetworkVisualizer;

export class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private selected_layer: NNLayer | null = null;
    private configuration_panel: HTMLElement;
    private configuration_content: HTMLElement;
    private placeholder: HTMLElement;

    constructor() {
        this.container = document.getElementById('network-container')!;
        this.configuration_panel = document.getElementById('configuration-panel')!;
        this.configuration_content = document.getElementById('configuration-content')!;
        this.placeholder = this.configuration_content.querySelector('.placeholder')!;

        document.addEventListener('click', (e)=> {
            // To ignore clicks inside the configuration panel
            if (this.configuration_panel.contains(e.target as Node)) {
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

    private createLayerElement(type:LayerType, neurons:number, activation:string): HTMLElement {
        const layer_element = document.createElement('div');
        layer_element.className = `layer ${type}`
        layer_element.textContent = `${type}\n${neurons}n\n${activation}`;
        layer_element.style.background = getLayerColor(type)
        layer_element.dataset.id = `layer-${Date.now()}`;
        
        return layer_element
    }

    private setupDocument() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges())
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer())
    }

    addLayer(type: LayerType, neurons: number = type === 'output' ? 1 : 64, activation: string = type === 'output' ? 'sigmoid': 'relu') {
        // check if the last added layer was an output layer
        if (type === 'output') neurons = 1;
        const last_layer = this.layers[this.layers.length-1];
        if(last_layer?.type === 'output') {
            alert("Cannot add layers after the output layer. If you want to add more layers, please remove the output layer first");
            return;
        }

        const layer_element = this.createLayerElement(type, neurons, activation)

        const new_layer: NNLayer = {
            id: `layer-${Date.now()}`,
            type,
            neurons,
            activation: activation,
            element: layer_element
        };

        this.layers.push(new_layer)
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

        this.selected_layer = layer;

        this.layers.forEach(l=>l.element.classList.remove('selected'));
        layer.element.classList.add('selected');

        this.updateConfigurationPanel(layer);
    }

    private deselectLayer() {
        this.selected_layer = null;
        this.layers.forEach(l=>l.element.classList.remove('selected'));

        this.placeholder.style.display = 'block';
        this.configuration_content.querySelectorAll('.layer-info, .form-group, button').forEach(el=> {
            (el as HTMLElement).style.display = 'none';
        })
    }

    private updateConfigurationPanel(layer: NNLayer) {
        const neurons_input = document.getElementById('neurons-input') as HTMLInputElement;
        const activation_select = document.getElementById('activation-select') as HTMLSelectElement;

        const layer_position = document.getElementById('layer-position') as HTMLElement;
        const layer_type = document.getElementById('layer-type') as HTMLElement;

        // Update layer info
        const layer_index = this.layers.findIndex(l => l.id === layer.id);
        layer_position.textContent = `${layer_index + 1}`
        layer_type.textContent = layer.type;
        
        // Update neurons and activation
        neurons_input.value = layer.neurons.toString();
        activation_select.value = layer.activation;
        
        // Hide placeholder and show configuration content
        this.placeholder.style.display = 'none';
        this.configuration_content.querySelectorAll('.layer-info, .form-group, button').forEach(el => {
            (el as HTMLElement).style.display = 'block';
        });
    }

    private applyLayerChanges() {
        if (!this.selected_layer) return;

        const neurons = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
        const activation = (document.getElementById('activation-select') as HTMLSelectElement).value;

        this.selected_layer.neurons = neurons;
        this.selected_layer.activation = activation;
        this.selected_layer.element.textContent = `${this.selected_layer.type}\n${neurons}n \n${activation}`;
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

async function handleRunClick() {
    if (getIsTraining()) {
        console.log("Stop button clicked: Requesting stop")
        requestStopTraining();
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

    const model = createEngineModelFromVisualizer(VISUALIZER, nin);

    startTraining();
    if (runStopButton) runStopButton.textContent = 'Stop training';

    try {
        await trainModel(model, train_x, train_y, crossEntropyLoss, LEARNING_RATE, ITERATIONS, LOG_FREQ)
    } catch (error) {
        console.error("Training failed:", error);
        throw error;
    } finally {
        endTraining();
        if (runStopButton) runStopButton.textContent = 'Run Model';
    }

    console.log(model);
}
