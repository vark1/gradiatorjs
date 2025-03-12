// import { DATASET_HDF5_TRAIN, DATASET_HDF5_TEST, prepare_dataset } from "utils/utils_data";

interface NNLayer {
    id: string;
    type: LayerType;
    position: {x: number, y: number}
    size: {width: number, height: number}
    neurons: number
    activation: string
}

interface Connection {
    from: string;
    to: string;
}

type LayerType = 'input' | 'dense' | 'conv' | 'output';

class NeuralNetworkVisualizer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private layers: NNLayer[] = [];
    private connections: Connection[] = [];
    private selected_element: {type: 'layer' | 'connection', id: string} | null = null;
    private drag_start_pos = {x:0, y:0};

    constructor(canvas_id: string) {
        this.canvas = document.getElementById(canvas_id) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;

        this.setupEventListeners();
        this.setupToolbar();
    }

    private setupEventListeners(){
        // arrow fn is to ensure 'this' is class instance and not the canvas element. definitely one of the languages of all time
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e))
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e))
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e))
        this.canvas.addEventListener('contextmenu', (e) => this.handleRightClick(e))
    }

    private setupToolbar() {
        document.getElementById('add-dense')?.addEventListener('click', ()=> this.addLayer('dense'));
        document.getElementById('add-conv')?.addEventListener('click', ()=> this.addLayer('conv'));
        document.getElementById('add-output')?.addEventListener('click', ()=> this.addLayer('output'));
    }

    public addLayer(type: LayerType, neurons: number = 64) {
        const new_layer: NNLayer = {
            id: `layer-${Date.now()}`,
            type,
            position: {x: 100 + this.layers.length * 150, y: 200},
            size: {width: 70, height: 20 + neurons*2},
            neurons,
            activation: 'relu'
        };
        this.layers.push(new_layer)
        this.draw();
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

    private draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        //drawing the connection
        this.connections.forEach(conn => {
            const from_layer = this.layers.find(l => l.id === conn.from);
            const to_layer = this.layers.find(l => l.id === conn.to);

            if(from_layer && to_layer) {
                this.ctx.beginPath();
                this.ctx.moveTo(from_layer.position.x + from_layer.size.width, from_layer.position.y + from_layer.size.height/2);
                this.ctx.lineTo(to_layer.position.x, to_layer.position.y + to_layer.size.height/2);
                this.ctx.strokeStyle='#666';
                this.ctx.lineWidth=2;
                this.ctx.stroke();
            }
        });

        //drawing the layers
        this.layers.forEach(layer => {
            this.ctx.fillStyle = this.getLayerColor(layer.type);
            this.ctx.fillRect(
                layer.position.x,
                layer.position.y,
                layer.size.width,
                layer.size.height
            );

            //Drawing neurons
            const neuron_spacing = layer.size.height/(layer.neurons+1);
            for(let i=1; i<=layer.neurons; i++) {
                this.ctx.beginPath();
                this.ctx.arc(
                    layer.position.x + layer.size.width,
                    layer.position.y + neuron_spacing * i,
                    3, 0, Math.PI*2
                );
                this.ctx.fillStyle='#fff'
                this.ctx.fill();
            }
        });
    }

    private handleMouseDown(e: MouseEvent) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        //check layer selection
        const clickedLayer = this.layers.find(layer=>
            x>layer.position.x &&
            x<layer.position.x + layer.size.width &&
            y>layer.position.y &&
            y<layer.position.y + layer.size.height
        );

        if(clickedLayer) {
            this.selected_element = {type: 'layer', id: clickedLayer.id};
            this.drag_start_pos = {x, y};
        }
    }

    //handle mouse movement when something is selected
    private handleMouseMove(e: MouseEvent) {
        if (this.selected_element?.type === 'layer') {  
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const layer = this.layers.find(l => l.id === this.selected_element!.id);
            if (layer) {
                layer.position.x += x - this.drag_start_pos.x;
                layer.position.y += y - this.drag_start_pos.y;
                this.drag_start_pos = {x, y};
                this.draw();
            }
        }
    }

    private handleMouseUp(e: MouseEvent) {
        this.selected_element = null;
    }

    private handleRightClick(e: MouseEvent) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        //find source layer
        const source_layer = this.layers.find(layer =>
            x > layer.position.x + layer.size.width - 20 &&
            x < layer.position.x + layer.size.width && 
            y > layer.position.y &&
            y < layer.position.y + layer.size.height
        );

        console.log(source_layer)

        if (source_layer) {
            //find target layer
            const target_layer = this.layers.find(layer=>
                x > layer.position.x &&
                x < layer.position.x + 20 &&
                y > layer.position.y && 
                y < layer.position.y + layer.size.height
            );

            if(target_layer && source_layer.id !== target_layer.id) {
                this.connections.push({
                    from: source_layer.id,
                    to: target_layer.id
                });
                this.draw();
            }
        }
    }

    private validateConnections(layers: NNLayer[]) {
        return true
    }

    prepareNetworkConfig() {
        const layers: number[] = [];
        const activations: string[] = [];

        //sort layers based on positions
        const sorted_layers = [...this.layers].sort((a, b) => a.position.x - b.position.x)

        //validate connections
        const valid_connections = this.validateConnections(sorted_layers);

        sorted_layers.forEach((layer, index) => {
            layers.push(layer.neurons);
            if(index < sorted_layers.length - 1) {
                activations.push(layer.activation);
            }
        });

        return {
            layers,
            activations,
            learning_rate: 0.0075
        }
    }
}

document.addEventListener('DOMContentLoaded', ()=> {
    const visualizer = new NeuralNetworkVisualizer('nn-canvas')
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