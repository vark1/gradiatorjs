import { assert } from '../utils/utils.js';
import { Val } from './val.js';

export function relu (Z: Val) : Val{
    let out = new Val(Z.shape)
    out.data = Z.data.map((k:number)=> Math.max(k, 0));

    out._prev = new Set([Z]);
    out._backward = () => {
        for(let i=0; i<Z.size; ++i) {
            Z.grad[i] += (Z.data[i] > 0 ? out.grad[i] : 0);
        }
    }
    return out
}

export function sigmoid (Z: Val) : Val {
    let out = new Val(Z.shape)
    out.data = Z.data.map((k:number)=>(1 / (1 + Math.exp(-k))));

    out._prev = new Set([Z]);
    out._backward = () => {
        for (let i=0; i<Z.size; ++i) {
            Z.grad[i] += out.data[i] * (1-out.data[i]) * out.grad[i];
        }
    }
    return out;
}

export function tanh (Z: Val) : Val{
    let out = new Val(Z.shape);
    out.data = Z.data.map((k:number)=>(Math.tanh(k)));

    out._prev = new Set([Z]);
    out._backward = () => {
        for (let i=0; i<Z.size; ++i) {
            Z.grad[i] += (1 - out.data[i] * out.data[i]) * out.grad[i];
        }
    }
    return out
}

export function softmax (Z: Val): Val {

    assert(Z.dim === 2, () => `Softmax: input must be 2D ([Batch, Classes]). Got ${Z.dim}D.`);

    return Z;   // returning this rn because we now have a softmax+crossentropy mix function so softmax by itself is not needed
}