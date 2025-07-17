import { Val } from "./val.js";
import { Module } from "./layers.js";

export class Sequential extends Module {
    layers: Module[];

    constructor(...layers: Module[]) {
        super();
        this.layers = layers;
    }

    override forward(X: Val) : Val {
        let currentOutput = X;
        for (const layer of this.layers) {
            currentOutput = layer.forward(currentOutput);
        }
        this.last_A = currentOutput;
        return currentOutput;
    }

    // Performs a forward pass and returns the intermediate pre- and post-activation
    // outputs of each layer in the sequence. 
    getLayerOutputs(X: Val): {Z: Val|null, A: Val|null}[] {
        this.forward(X);
        const outputs = this.layers.map(layer => ({
            Z: layer.last_Z,
            A: layer.last_A
        }));
        return outputs;
    }

    override toJSON(): any {
        const modelJSON = {
            modelType: 'Sequential',
            layers: this.layers.map(layer => layer.toJSON())
        };
        return modelJSON;
    }
}