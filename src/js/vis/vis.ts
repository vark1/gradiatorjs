// import { DATASET_HDF5_TRAIN, DATASET_HDF5_TEST, prepare_dataset } from "utils/utils_data";
import { getLayerColor, LayerType, NNLayer, NetworkConfig } from "./vis_utils.js";

class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private context_menu: HTMLElement;
    private selected_layer: NNLayer | null = null;

    constructor() {
        this.container = document.getElementById('network-container')!;
        this.context_menu = document.getElementById('context-menu')!;

        this.setupToolbar();
        this.setupContextmenu();
        
        // hide context menu when clicking outside
        document.addEventListener('click', (e)=> {
            if (!this.context_menu.contains(e.target as Node)) this.hideContextMenu();
        });

        // hide context menu when right-clicking outside layers
        document.addEventListener('contextmenu', (e)=> {
            if(!(e.target as HTMLElement).closest('.layer')) this.hideContextMenu();
        });
    }

    private setupToolbar() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
    }

    private setupContextmenu() {
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges())
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer())
    }

    addLayer(type: LayerType, neurons: number = 64, activation: string = 'relu') {
        // check if the last added layer was an output layer
        const last_layer = this.layers[this.layers.length-1];
        if(last_layer?.type === 'output') {
            alert("Cannot add layers after the output layer. If you want to add more layers, please remove the output layer first");
            return;
        }

        const layer_element = document.createElement('div');
        layer_element.className = `layer ${type}`
        layer_element.textContent = `${type}\n${neurons}n\n${activation}`;
        layer_element.style.background = getLayerColor(type)
        layer_element.dataset.id = `layer-${Date.now()}`;

        const new_layer: NNLayer = {
            id: `layer-${Date.now()}`,
            type,
            neurons,
            activation: activation,
            element: layer_element
        };

        layer_element.addEventListener('contextmenu', (e)=> {
            e.preventDefault();
            this.showContextMenu(e, new_layer);
        });

        this.layers.push(new_layer)
        this.container.appendChild(layer_element);

    }

    deleteSelectedLayer() {
        if (!this.selected_layer) return;

        if(confirm('Are you sure you want to delete this layer?')) {
            this.removeLayer(this.selected_layer.id);
            this.hideContextMenu();
        }
    }

    private removeLayer(layer_id: string) {
        const layer_index = this.layers.findIndex(layer => layer.id === layer_id);
        if (layer_index === -1) return; //layer not found

        const layer_element = this.layers[layer_index].element;
        this.container.removeChild(layer_element);

        this.layers.splice(layer_index, 1);
    }

    private showContextMenu(e: MouseEvent, layer: NNLayer) {
        e.preventDefault();
        this.selected_layer = layer;

        // disabling neuron count for output layer
        const neurons_input = document.getElementById('neurons-input') as HTMLInputElement;
        neurons_input.disabled = layer.type === 'output';

        (document.getElementById('neurons-input') as HTMLInputElement).value = layer.neurons.toString();
        (document.getElementById('activation-select') as HTMLSelectElement).value = layer.activation;

        this.context_menu.style.display = 'block'
        this.context_menu.style.left = `${e.pageX}px`;
        this.context_menu.style.top = `${e.pageY}px`;
    }

    private hideContextMenu() {
        this.context_menu.style.display = 'none';
        this.selected_layer = null;
    }

    private applyLayerChanges() {
        if (!this.selected_layer) return;

        const neurons = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
        const activation = (document.getElementById('activation-select') as HTMLSelectElement).value;

        this.selected_layer.neurons = neurons;
        this.selected_layer.activation = activation;
        this.selected_layer.element.textContent = `${this.selected_layer.type}\n${neurons}n \n${activation}`;

        this.hideContextMenu();
    }

    getNetworkConfig(): NetworkConfig {
        return {
            layer_sizes: this.layers.map(l=>l.neurons),
            activations: this.layers.map(l=>l.activation)
        }
    }

    validateNetwork(): string[] {
        const errors: string[] = [];
        if (this.layers.length<2)   errors.push("Need atleast 2 layers");
        if (!this.layers.some(l=>l.type === 'output')) errors.push("Missing output layer");
        return errors;
    }
}

document.addEventListener('DOMContentLoaded', ()=> {
    const visualizer = new NeuralNetworkVisualizer()
    console.log(visualizer)
})

const button = document.getElementById('run_model_btn')
// if(button) {
//     button.addEventListener('click', function() {
//         if(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN) {
//             const [train_x, train_y, test_x, test_y] = prepare_dataset();
//             const network_config = visualizer.prepareNetworkConfig();

//             const L_layer_model = MLP(
//                 train_x,
//                 train_y,
//                 network_config.layers,
//                 network_config.learning_rate,
//                 2500,
//                 true
//             );
//             console.log(L_layer_model)
//         }
//     });
// }




