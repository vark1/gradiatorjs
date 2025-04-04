import { Val } from "../Val/val.js"
import * as op from "../Val/ops.js"
import * as afn from "../Val/activations.js"
import { gaussianRandom } from "utils/utils_num.js";

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

    constructor (nin: number, nout: number, activation?:(t: Val) => Val) {
        super();

        this.W = new Val([nin, nout]);
        this.W.data = this.W.data.map(()=>gaussianRandom() * Math.sqrt(2.0)/nin);

        this.B = new Val([1, nout], 0.1);
        this.activation = activation
    }

    forward(X: Val) : Val {
        // X.shape = [batchsize, nin]
        let X_ = X;
        if (X.dim === 1) {
            // reshaping [nin] to [1, nin]
            X_ = X.reshape([1, X.shape[0]]);
        } else if(X.dim !== 2) {
            throw new Error(`Dense layer expects dim 1 or 2, got ${X.dim}`);
        }

        if (X_.shape[1] !== this.W.shape[0]) {
            throw new Error(`Input features ${X_.shape[1]} don't match layer input size ${this.W.shape[0]}`)
        }

        const Z = op.add(op.dot(X_, this.W), this.B);

        const A = this.activation ? this.activation(Z) : Z;

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
}

function loss(w: Val, b: Val, X: Val, Y: Val) {
    let m = X.shape[1]
    let z = op.add(op.dot(w.T, X), b)
    let A = afn.sigmoid(z)
    // cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
    let cost = op.mul(-1/m, op.sum(op.add(op.mul(Y,op.log(A)), op.mul(op.sub(1, Y), op.log(op.sub(1, A))))))

}