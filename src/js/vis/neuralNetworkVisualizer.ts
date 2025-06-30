import { SerializableNNLayer, LayerCreationOptions, DatasetOption } from "../types_and_interfaces/vis_interfaces.js";
import { ActivationType, LayerType, NNLayer } from "../types_and_interfaces/general.js";
import { getLayerColor } from "../utils/utils_vis.js";
import { PRESETS } from "../utils/utils_vis_preset.js";

export class NeuralNetworkVisualizer {
    private container: HTMLElement;
    private layers: NNLayer[] = [];
    private selected_layer: NNLayer | null = null;
    private config_panel: HTMLElement;
    private placeholder: HTMLElement;
    private draggedLayerId: string | null = null;

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
        this.setupDragAndDrop();
        this.loadNetworkFromLocalStorage();
    }

    private renderNetwork(): void {
        this.container.innerHTML = '';
        this.layers.forEach((layer, idx)=> {
            if (idx>0) {
                const arrow = document.createElement('div');
                arrow.className = 'layer-connector';
                this.container.appendChild(arrow);
            }
            const layerEl = this.createLayerElement(layer);
            layer.element = layerEl;
            this.container.appendChild(layerEl);
        });

        if (this.selected_layer) {
            const selectedEl = this.layers.find(l=>l.id===this.selected_layer!.id)?.element;
            if (selectedEl) {
                selectedEl.classList.add('selected');
            }
        };
    }

    private setupEventListeners() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer({type: 'dense', neurons: 2, activation: 'relu'}));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer({type: 'conv', out_channels: 8, kernel_size: 5, stride: 2, padding: 2, activation: 'relu'}));
        document.getElementById('add-flatten')?.addEventListener('click', ()=> this.addLayer({type:'flatten'}));
        document.getElementById('add-maxpool')?.addEventListener('click', ()=> this.addLayer({type:'maxpool', pool_size: 2, stride: 2}));
        document.getElementById('apply-layer-changes')?.addEventListener('click', ()=>this.applyLayerChanges());
        document.getElementById('delete-selected-layer')?.addEventListener('click', ()=>this.deleteSelectedLayer());
        document.getElementById('dataset-select')?.addEventListener('change', (e) => {
            const value = (e.target as HTMLSelectElement).value as DatasetOption;
            this.applyDatasetPreset(value);
        });

        document.getElementById('clear-network')?.addEventListener('click', ()=> {
            if (!confirm("Are you sure you want to clear the current network?")) return;
            this.clearNetwork();
            const statusDiv = document.getElementById('persistence-status');
            if (!statusDiv) return;
            statusDiv.textContent = 'Current network cleared.';
            setTimeout(()=> { statusDiv.textContent = '';}, 2000);
        });
        
        document.getElementById('save-network')?.addEventListener('click', ()=> {
            this.saveNetworkToLocalStorage();
            const statusDiv = document.getElementById('persistence-status');
            if (!statusDiv) return;
            statusDiv.textContent = 'Network saved to browser storage.';
            setTimeout(()=> { statusDiv.textContent = '';}, 2000);
        });

        document.getElementById('load-network')?.addEventListener('click', ()=> {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                alert("No saved network found in browser storage.");
                return;
            }

            const userConfirmed = confirm("Loading a saved network will overwrite your current progress. Are you sure you want to continue?");
            
            if (userConfirmed) {
                this.loadNetworkFromLocalStorage();
            } else {
                console.log("load operation cancelled by the user.")
            }
        });
        
        document.getElementById('clear-saved-network')?.addEventListener('click', ()=> {
            localStorage.removeItem(this.STORAGE_KEY);
            const statusDiv = document.getElementById('persistence-status');
            if (statusDiv) statusDiv.textContent = 'Cleared saved network from browser storage.';
            console.log("Cleared saved network.");
        });
    }
    
    private setupDragAndDrop(): void {
        this.container.addEventListener('dragstart', (e)=> {
            const target = e.target as HTMLElement;
            if (target.classList.contains('layer')) {
                this.draggedLayerId = target.dataset.id || null;
                setTimeout(()=>target.classList.add('dragging'), 0);
            }
        });
        
        this.container.addEventListener('dragend', (e)=> {
            const target = e.target as HTMLElement;
            if (target.classList.contains('layer')) {
                target.classList.remove('dragging');
            }
            this.draggedLayerId = null;
        });

        this.container.addEventListener('dragover', (e)=> {
            e.preventDefault();
            const placeholder = this.getOrCreatePlaceholder();
            const afterEl = this.getDragAfterEl(this.container, e.clientX);
            this.container.insertBefore(placeholder, afterEl);
        });

        this.container.addEventListener('drop', (e)=> {
            e.preventDefault();
            this.container.querySelector('.drag-over-placeholder')?.remove();
            if (!this.draggedLayerId) return;

            const draggedIdx = this.layers.findIndex(l=>l.id===this.draggedLayerId);
            if (draggedIdx===-1)    return;
            
            const draggedLayer = this.layers.splice(draggedIdx, 1)[0];
            const afterEl = this.getDragAfterEl(this.container, e.clientX);
            const afterElId = afterEl ? (afterEl as HTMLElement).dataset.id : null;
            const newIdx = afterElId ? this.layers.findIndex(l=>l.id===afterElId):this.layers.length;

            this.layers.splice(newIdx, 0, draggedLayer);
            this.renderNetwork();
        });
    }

    private getOrCreatePlaceholder(): HTMLElement {
        let placeholder = this.container.querySelector('.drag-over-placeholder');
        if (!placeholder) {
            placeholder = document.createElement('div');
            placeholder.className = 'drag-over-placeholder';
        }
        return placeholder as HTMLElement;
    }

    getDragAfterEl(container: HTMLElement, x: number) {
        const draggableEl = Array.from(container.querySelectorAll('.layer:not(.dragging)'));
        return draggableEl.reduce((closest: { offset: number, element: Element | null }, child) => {
            const box = child.getBoundingClientRect();
            const offset = x-box.left-box.width/2;
            if (offset<0 && offset>closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY, element: null }).element;
    }

    applyDatasetPreset(datasetName: DatasetOption) {
        const preset = PRESETS[datasetName];
        if (!preset) return;

        if (!confirm(`Do you want to replace your current network and params(l_rate, batchsize, epochs) with the preset for "${datasetName}"?`)) return;
        this.clearNetwork();
        preset.network.forEach(opts => this.addLayer(opts));

        (<HTMLInputElement>document.getElementById('learning-rate')).value = preset.params.l_rate.toString();
        (<HTMLInputElement>document.getElementById('epoch')).value = preset.params.epochs.toString();
        (<HTMLInputElement>document.getElementById('batch-size')).value = preset.params.batch_size.toString();
        console.log(`Loaded preset for ${datasetName}`);
    }

    // Converts the current network layer structure into a serializable JSON object.
    private getNetworkConfig(): SerializableNNLayer[] {
        return this.layers.map(layer => {
            const { element, ...serializableLayer} = layer;
            return serializableLayer;
        })
    }

    private clearNetwork() {
        this.layers = [];
        this.selected_layer = null;
        if (this.container) this.container.innerHTML = '';
    }

    private loadNetworkFromLocalStorage(): void {
        const statusDiv = document.getElementById('persistence-status');
        
        const jsonString = localStorage.getItem(this.STORAGE_KEY);
        if (!jsonString) {
            console.log("No saved network found in localStorage.");
            if (statusDiv) statusDiv.textContent = 'No saved network found.';
            return;
        }
        try {
            const savedConfig: SerializableNNLayer[] = JSON.parse(jsonString);
            if (!Array.isArray(savedConfig)) {
                throw new Error("Saved data is not a valid network configuration.");
            }

            this.clearNetwork();
            
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
        layer_element.setAttribute('draggable', 'true');
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
                newLayer = { id, type: options.type, neurons: options.neurons, activation: options.activation, element: null as any};
                break;
            case 'conv':
                newLayer = { id, type: options.type, out_channels: options.out_channels, kernel_size: options.kernel_size, stride: options.stride, padding: options.padding, activation: options.activation, element: null as any };
                break;
            case 'flatten':
                newLayer = { id, type: options.type, element: null as any };
                break;
            case "maxpool":
                newLayer = { id, type: options.type, pool_size: options.pool_size, stride: options.stride, element: null as any, };
                break;
            default:
                console.error("Unknown layer type for options:", options);
                return;
        }

        if (this.container.children.length > 0) {
            const arrow = document.createElement('div');
            arrow.className = 'layer-connector';
            arrow.dataset.arrowFor = id;
            this.container.appendChild(arrow);
        }
        const layer_element = this.createLayerElement(newLayer)
        newLayer.element = layer_element;

        this.container.appendChild(layer_element);

        this.layers.push(newLayer)
        this.container.appendChild(layer_element);

        this.renderNetwork();
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

        const precedingElement = layer_element.previousElementSibling;
        if (precedingElement && precedingElement.classList.contains('layer-connector')) {
            this.container.removeChild(precedingElement);
        }

        this.container.removeChild(layer_element);

        this.layers.splice(layer_index, 1);

        this.renderNetwork();
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