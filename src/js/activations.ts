import {add, sub, pow, mul} from './Val/ops.js'
import { Val } from './Val/val.js';

const ACT_MAP = {
    "relu": (x: number) => Math.max(x, 0),
    "sigmoid": (x: number) => 1 / (1 + Math.exp(-x)),
    "tanh": (x: number) => Math.tanh(x)
};

const DER_MAP = {
    "relu" : function (x: Val) {
        let y = new Val(x.shape)
        y.data = x.data.map((v: number) => (v === 0 ? 0 : 1))
        return y
    },
    "sigmoid": (x: Val) => mul(x, sub(1, x)),
    "tanh": (x: Val) => sub(1, pow(x, 2))
}

export function activationfn(t: Val, type: keyof typeof ACT_MAP = 'relu'): Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k: number) => ACT_MAP[type](k))
    
    // Backward pass using derivative
    // out._backward = () => {
    //     let derivative = mul(DER_MAP[type](data), out.grad);
    //     t_.grad = add(t_.grad, derivative).data;
    // };

    return x;
}