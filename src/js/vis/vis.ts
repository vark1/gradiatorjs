// import { DATASET_HDF5_TRAIN, DATASET_HDF5_TEST, prepare_dataset } from "utils/utils_data";

interface NNLayer {
    id: string;
    type: LayerType;
    neurons: number;
    activation: string;
    element: HTMLElement;
}

type LayerType = 'input' | 'dense' | 'conv' | 'output';

class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private context_menu: HTMLElement;
    private selected_layer: NNLayer | null = null;

    constructor() {
        this.container = document.getElementById('network-container')!;
        this.context_menu = document.getElementById('context-menu')!;

        this.setupToolbar();

        document.addEventListener('click', ()=> this.hideContextMenu());
        document.onclick
        document.addEventListener('contextmenu', (e)=> {
            if(!(e.target as HTMLElement).closest('.layer')) this.hideContextMenu();
        });
    }

    private getLayerColor(type: LayerType): string {
        const colors = {
            input: '#4CAF50',
            dense: '#2196F3',
            conv: '#FF9800',
            output: '#F44336'
        };
        return colors[type] || '#999';
    }

    private setupToolbar() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
    }

    addLayer(type: LayerType, neurons: number = 64) {
        const layer_element = document.createElement('div');
        layer_element.className = `layer ${type}`
        layer_element.textContent = `${type}\n${neurons}n`;
        
        layer_element.style.background = this.getLayerColor(type)

        const new_layer: NNLayer = {
            id: `layer-${Date.now()}`,
            type,
            neurons,
            activation: 'relu',
            element: layer_element
        };

        layer_element.addEventListener('contextmenu', (e)=> {
            e.preventDefault();
            this.showContextMenu(e, new_layer);
        });

        this.layers.push(new_layer)
        this.container.appendChild(layer_element);
    }

    private showContextMenu(e: MouseEvent, layer: NNLayer) {
        e.preventDefault();
        this.selected_layer = layer;

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

    applyLayerChanges() {
        if (!this.selected_layer) return;

        const neurons = parseInt((document.getElementById('neurons-input') as HTMLInputElement).value);
        const activation = (document.getElementById('activation-select') as HTMLSelectElement).value;

        this.selected_layer.neurons = neurons;
        this.selected_layer.activation = activation;
        this.selected_layer.element.textContent = `${this.selected_layer.type}\n${neurons}n \n${activation}`;

        this.hideContextMenu();
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