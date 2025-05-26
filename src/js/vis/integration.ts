import { Sequential, Dense, Module, Conv } from '../nn/nn.js';
import * as afn from '../Val/activations.js';
import { NeuralNetworkVisualizer } from './vis.js';
import { LayerType, NNLayer } from '../utils/utils_vis.js';

const AFN_MAP = {
    "relu": afn.relu,
    "sigmoid": afn.sigmoid,
    "tanh": afn.tanh
}

export function createEngineModelFromVisualizer(
    visualizer: NeuralNetworkVisualizer,
    nin: number,
    inH?: number, // if first layer is conv
    inW?: number // if first layer is conv
): Sequential {

    const errors = visualizer.validateNetwork();
    if (errors.length > 0) {
        throw new Error(`Invalid network configuration: ${errors.join(', ')}`);
    }

    const vizLayers: NNLayer[] = (visualizer as any).layers;

    if (!vizLayers || vizLayers.length === 0) {
        throw new Error("Visualizer has no layers defined.");
    }

    const engineLayers: Module[] = [];
    let currentInputUnits = nin;
    let currentH = inH;
    let currentW = inW;
    let prevLayerType: LayerType = vizLayers[0].type;

    for (const vizLayer of vizLayers) {
        const activationFn = AFN_MAP[vizLayer.activation as keyof typeof AFN_MAP];
        let engineLayer: Module;

        switch (vizLayer.type) {
            case 'dense':
                let ninForDense = currentInputUnits;

                if (prevLayerType === 'conv') {
                    if (currentH === undefined || currentW === undefined) throw new Error(`cannot create dense layer after conv since output dims of prev layer are unknown`);
                    ninForDense = currentH * currentW * currentInputUnits;
                }

                engineLayer = new Dense(
                    ninForDense,
                    vizLayer.neurons,
                    activationFn
                );
                engineLayers.push(engineLayer);
                currentInputUnits = vizLayer.neurons; // Output size of this layer is input size for next
                currentW = undefined;
                currentH = undefined;
                break;

            case 'conv':
                if (prevLayerType === 'dense') {
                    console.warn(`Creating Conv layer after Dense layer. Assuming Dense output represents [Batch, 1, 1, ${currentInputUnits}] or needs explicit Reshape. Setting H_in=1, W_in=1 for Conv.`);
                    currentH = 1;
                    currentW = 1;
                }

                if (currentH === undefined || currentW === undefined || currentInputUnits === undefined) {
                    throw new Error(`Cant create Conv layer: input dims (H, W, C) are unknown or undefined. H: ${currentH}, W: ${currentW}, C_in: ${currentInputUnits}. Previous layer type: ${prevLayerType}`)
                }

                engineLayer = new Conv(
                    currentInputUnits, 
                    vizLayer.out_channels, 
                    vizLayer.kernel_size, 
                    vizLayer.stride, 
                    vizLayer.padding, 
                    activationFn
                );
                engineLayers.push(engineLayer);

                currentH = Math.floor((currentH - vizLayer.kernel_size + 2 * vizLayer.padding) / vizLayer.stride) + 1;
                currentW = Math.floor((currentW - vizLayer.kernel_size + 2 * vizLayer.padding) / vizLayer.stride) + 1;
                if (currentH <= 0 || currentW <= 0) {
                    throw new Error(`Conv layer results in non-positive output dimension: H_out=${currentH}, W_out=${currentW}. Check params.`);
                }

                currentInputUnits = vizLayer.out_channels;
                break;
            case 'output':      // this is also a dense layer, just the last one (likely with 1-10 neurons only)
                let ninForOutput = currentInputUnits;

                if (prevLayerType === 'conv') {
                    if (currentH === undefined || currentW === undefined) throw new Error(`cannot create dense layer after conv since output dims of prev layer are unknown`);
                    ninForOutput = currentH * currentW * currentInputUnits;
                }

                engineLayer = new Dense(
                    ninForOutput,
                    vizLayer.neurons,
                    activationFn
                );
                engineLayers.push(engineLayer);
                currentInputUnits = vizLayer.neurons; // Output size of this layer is input size for next
                currentW = undefined;
                currentH = undefined;
                break;    
            default:
                throw new Error(`Unsupported visualizer layer type for layer: ${vizLayer}`);        
        }
        prevLayerType = vizLayer.type;
    }

    if (engineLayers.length === 0) {
        throw new Error("No computational engine layers could be created from the visualizer configuration.");
    }

    const model = new Sequential(...engineLayers);
    return model;
}