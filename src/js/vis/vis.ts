// import { DATASET_HDF5_TRAIN, DATASET_HDF5_TEST, prepare_dataset } from "utils/utils_data";
import { getLayerColor, LayerType, NNLayer, NetworkConfig } from "./vis_utils.js";

class NeuralNetworkVisualizer {
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
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
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
        this.configuration_content.querySelectorAll('.form-group, button').forEach(el=> {
            (el as HTMLElement).style.display = 'none'
        })
    }

    private updateConfigurationPanel(layer: NNLayer) {
        const neurons_input = document.getElementById('neurons-input') as HTMLInputElement;
        const activation_select = document.getElementById('activation-select') as HTMLSelectElement
        
        neurons_input.value = layer.neurons.toString();
        neurons_input.disabled = layer.type === 'output';
        activation_select.value = layer.activation;
        
        this.placeholder.style.display = 'none';
        this.configuration_content.querySelectorAll('.form-group, button').forEach(el => {
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




