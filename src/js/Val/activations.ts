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

    let out = new Val(Z.shape);
    const batchsize = Z.shape[0]
    const classes = Z.shape[1]

    for (let b=0; b<batchsize; b++) {
        let max_logit = -Infinity;
        for (let j=0; j<classes; j++) {
            if (Z.data[b*classes+j] > max_logit) {
                max_logit = Z.data[b*classes+j];
            }
        }

        let sum_exps = 0;   // calc exponents and their sum
        for (let j=0; j<classes; j++) {
            const exp_val = Math.exp(Z.data[b*classes+j]-max_logit);
            out.data[b*classes+j] = exp_val;
            sum_exps += exp_val
        }

        for (let j=0; j<classes; j++) {
            out.data[b*classes+j]/=sum_exps;    //normalize to get probs
        }
    }

    out._prev = new Set([Z]);
    out._backward = () => {
        // The gradient dL/dz for a single sample is:
        // p * (dL/dp - sum(dL/dp_i * p_i))
        // where 
        // p        -> output probability vector 
        // dL/dp    -> incoming gradient.

        if (!Z.grad || Z.grad.length !== Z.size) {
            Z.grad = new Float64Array(Z.size).fill(0);
        }

        // jacobian matrix
        for (let b=0; b<batchsize; b++) {
            const rowOffset = b*classes;

            const p=out.data.subarray(rowOffset, rowOffset+classes);
            const dL_dp=out.data.subarray(rowOffset, rowOffset+classes);

            let dot=0;  // dot product: sum(dL/dp_i * p_i)
            for (let j=0; j<classes; j++) {
                dot+=dL_dp[j]*p[j];
            }

            // calculating grad dL/dz for thsi row. should be: dL/dz_j = p_j * (dL/dp_j - dot_product)
            for (let j=0; j<classes; j++) {
                const grad = p[j]*(dL_dp[j]-dot);
                Z.grad[rowOffset+j] += grad;
            }
        }
    }

    return out;
}