import { Val } from "../Val/val.js"
import * as op from "../Val/ops.js"
import { gaussianRandom } from "../utils/utils_num.js";
import { assert } from "../utils/utils.js";

/*
A module represents a base class that will manage parameters or zero grads.

It will automatically find all the trainable Val instances within a potentially complex structure (model/layer)
*/
export class Module {
    parameters() : Val[] {
        let params: Val[] = [];
        for (const key in this) {
            const prop = this[key as keyof this];
            if (prop instanceof Val) {      // direct param of this module (eg, w, b in the DenseLayer)
                params.push(prop);
            } else if (prop instanceof Module) {        // nested modules (eg, custom layer containing other layers)
                params = params.concat(prop.parameters());
            } else if (Array.isArray(prop)) {       // Array of modules (like sequential)
                prop.forEach(item => {
                    if (item instanceof Module) {
                        params = params.concat(item.parameters())
                    } else if (item instanceof Val) {
                        params.push(item);
                    }
                });
            }
        }
        return [...new Set(params)];
    }

    zeroGrad() : void {
        this.parameters().forEach(p => p.grad.fill(0));
    }
}

/*Dense/fully connected layer (A = nonlin(Wt.X + B))*/
export class Dense extends Module {
    W: Val;
    B: Val;
    activation?: (t: Val) => Val;
    public readonly nin: number;
    public readonly nout: number;

    constructor (nin: number, nout: number, activation?:(t: Val) => Val) {
        super();
        this.nin = nin;
        this.nout = nout;

        this.W = new Val([nin, nout]);
        this.W.data = this.W.data.map(()=>gaussianRandom() * Math.sqrt(2.0/nin));

        this.B = new Val([1, nout], 0.1);
        this.activation = activation
    }

    forward(X_input: Val) : Val {
        // X.shape = [batchsize, nin]
        let X = X_input;

        // flattening for 4d input (like from conv layer)
        if (X_input.dim === 4) {
            const batchSize = X_input.shape[0];
            const flattenedFeatures = X_input.shape[1] * X_input.shape[2] * X_input.shape[3];

            console.warn(`Denselayer input is 4d. Flattening`)
            X = X_input.reshape([batchSize, flattenedFeatures])
        }

        // X should be either 1D or 2D
        if (X_input.dim === 1) {
            // reshaping [nin] to [1, nin]
            X = X_input.reshape([1, X_input.shape[0]]);
        } else if(X.dim !== 2) {
            throw new Error(`Dense layer expects dim 1, 2 or 4, got ${X_input.dim}`);
        }

        if (X.shape[1] !== this.nin) {
            throw new Error(`Input features ${X.shape[1]} don't match layer input size ${this.nin}`)
        }

        const Z = op.add(op.dot(X, this.W), this.B);
        const A = this.activation ? this.activation(Z) : Z;
        return A;
    }
}

export class Conv extends Module {
    kernel: Val;
    biases: Val;
    stride: number;
    padding: number;
    activation?: (t: Val) => Val;

    public readonly in_channels: number;
    public readonly out_channels: number;
    public readonly kernel_size: number;

    constructor(in_channels: number, out_channels: number, kernel_size: number, stride: number, padding: number, activation?:(t: Val) => Val) {
        super();
        
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.kernel_size = kernel_size;
        this.kernel = new Val([out_channels, kernel_size, kernel_size, in_channels])
        this.stride = stride
        this.padding = padding
        this.activation = activation

        const fan_in = in_channels * kernel_size * kernel_size;
        this.kernel.data = this.kernel.data.map(()=> gaussianRandom()*Math.sqrt(2.0/fan_in));

        const biasShape = [out_channels];
        this.biases = new Val(biasShape);
        this.biases.data.fill(0.1);
    }

    forward(X_input: Val): Val {
        let X = X_input;

        if (X_input.dim === 2 && X_input.shape[1] === this.in_channels) {
            console.warn(`Conv2DLayer: Input is 2D [${X_input.shape[0]}, ${X_input.shape[1]}]. Reshaping to 4D [${X_input.shape[0]}, 1, 1, ${this.in_channels}] assuming it came from a Dense layer.`);
            X = X_input.reshape([X_input.shape[0], 1, 1, this.in_channels]);
        }

        assert(X.dim === 4, ()=> `Conv2DLayer: Input must be 4D. got ${X.dim} dims with shape : ${X.shape}`);
        assert(X.shape[3] === this.in_channels, ()=> `Conv2DLayer: Input channels ${X.shape[3]} mismatch layer input channels ${this.in_channels}`)

        // NHWC Input with [C_out, kernelsize, kernelsize, C_in] weights
        // output should be [Batch, H_out, W_out, C_out]
        const Z_conv = op.conv2d(X, this.kernel, this.stride, this.padding);

        // TODO: manually adding bias here instead of using op.add because broadcasting is not yet available for 4D arrays
        const B_batch = X.shape[0];
        const H_out = Z_conv.shape[1];
        const W_out = Z_conv.shape[2];

        const Z_with_bias = new Val([B_batch, H_out, W_out, this.out_channels]);
        
        for (let b=0; b<B_batch; b++) {
            for (let h=0; h<H_out; h++) {
                for (let w=0; w<W_out; w++) {
                    for (let c=0; c<this.out_channels; c++) {
                        const z_conv_idx = b*(H_out*W_out*this.out_channels) + h*(W_out*this.out_channels) + w*this.out_channels + c;
                        Z_with_bias.data[z_conv_idx] = Z_conv.data[z_conv_idx] + this.biases.data[c];
                    }
                }
            }
        }

        Z_with_bias._prev = new Set([Z_conv, this.biases]);
        Z_with_bias._backward = () => {
            if (!Z_conv.grad || Z_conv.grad.length !== Z_conv.size) Z_conv.grad = new Float64Array(Z_conv.size).fill(0);
            for (let i = 0; i < Z_with_bias.grad.length; i++) {
                Z_conv.grad[i] += Z_with_bias.grad[i];
            }

            // Propagate gradient to biases: dL/dBias_c = sum(dL/dZ_with_bias over b, h, w for channel c)
            if (!this.biases.grad || this.biases.grad.length !== this.biases.size) this.biases.grad = new Float64Array(this.biases.size).fill(0);

            for (let b = 0; b < B_batch; b++) {
                for (let h = 0; h < H_out; h++) {
                    for (let w = 0; w < W_out; w++) {
                        for (let c = 0; c < this.out_channels; c++) {
                            const z_with_bias_grad_idx = b*(H_out*W_out*this.out_channels) + h*(W_out*this.out_channels) + w*this.out_channels + c;
                            this.biases.grad[c] += Z_with_bias.grad[z_with_bias_grad_idx];
                        }
                    }
                }
            }
        }

        const A = this.activation? this.activation(Z_with_bias) : Z_with_bias;
        return A;
    }
}

/*Sequential model (this is basically an MLP)*/
export class Sequential extends Module {
    layers: Module[];

    constructor(...layers: Module[]) {
        super();
        this.layers = layers;
    }

    forward(X: Val) : Val {
        let currentOutput = X;
        for (const layer of this.layers) {
            if (typeof (layer as any).forward === 'function') {
                currentOutput = (layer as any).forward(currentOutput);
            } else {
                throw new Error("Item in sequential model does not have a forward method.")
            }
        }
        return currentOutput;
    }

    getActivations(X: Val) : Val[] {
        let currentOutput = X;
        let activations : Val[] = [X];
        for (const layer of this.layers) {
            if (typeof (layer as any).forward === 'function') {
                currentOutput = (layer as any).forward(currentOutput);
                activations.push(currentOutput)
            } else {
                throw new Error("Item in sequential model does not have a forward method")
            }
        }
        return activations;
    }
}