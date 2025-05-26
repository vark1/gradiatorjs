import {assert, broadcast, reduceGradient} from '../utils/utils.js'
import { Val } from './val.js';

export function add(t1: Val|number, t2: Val|number) : Val {

    const originalT1 = (t1 instanceof Val) ? t1 : new Val([], t1 as number);
    const originalT2 = (t2 instanceof Val) ? t2 : new Val([], t2 as number);

    let [t1_, t2_] = broadcast(originalT1, originalT2);

    assert(t1_.dim === t2_.dim, ()=> `In addition: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In addition: Both matrices must have the same shape')

    let out = new Val(t1_.shape);
    out.data = t1_.data.map((num: number, idx: number) => num + t2_.data[idx]); // Forward
    out._prev = new Set([originalT1, originalT2]);
    out._backward = () => {
        const t1_grad = out.grad;                                               // Backward
        const t1_reduced_grad = reduceGradient(t1_grad, originalT1.shape, t1_.shape);
        originalT1.grad = originalT1.grad.map((g, i) => g + (t1_reduced_grad[i] || 0));

        const t2_grad = out.grad;
        const t2_reduced_grad = reduceGradient(t2_grad, originalT2.shape, t2_.shape);
        originalT2.grad = originalT2.grad.map((g, i) => g + (t2_reduced_grad[i] || 0));
    };
    return out;
}

export function sub(t1: Val|number, t2: Val|number) : Val {
    const originalT1 = (t1 instanceof Val) ? t1 : new Val([], t1 as number);
    const originalT2 = (t2 instanceof Val) ? t2 : new Val([], t2 as number);

    let [t1_, t2_] = broadcast(originalT1, originalT2);

    assert(t1_.dim === t2_.dim, ()=> `In subtraction: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In subtraction: Both matrices must have the same shape')

    let out = new Val(t1_.shape)
    out.data = t1_.data.map((num: number, idx: number)=>num - t2_.data[idx])
    out._prev = new Set([originalT1, originalT2])
    out._backward = () => {
        const t1_grad = out.grad;
        const t1_reduced_grad = reduceGradient(t1_grad, originalT1.shape, t1_.shape);
        originalT1.grad = originalT1.grad.map((g, i) => g + (t1_reduced_grad[i] || 0));

        const t2_grad = out.grad.map((v) => -v);
        const t2_reduced_grad = reduceGradient(t2_grad, originalT2.shape, t2_.shape);
        originalT2.grad = originalT2.grad.map((g, i) => g + (t2_reduced_grad[i] || 0));
    }
    return out;
}

export function mul(t1: Val|number, t2: Val|number) : Val {
    const originalT1 = (t1 instanceof Val) ? t1 : new Val([], t1 as number);
    const originalT2 = (t2 instanceof Val) ? t2 : new Val([], t2 as number);

    let [t1_, t2_] = broadcast(originalT1, originalT2);

    assert(t1_.dim === t2_.dim, ()=> `In hadamard product: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In hadamard product: Both matrices must have the same shape')

    let out = new Val(t1_.shape);
    out.data = t1_.data.map((num: number, idx: number)=> num*t2_.data[idx]);
    out._prev = new Set([originalT1, originalT2]);
    out._backward = () => {
        const t1_grad = out.grad.map((og, i) => og * t2_.data[i]);
        const t1_reduced_grad = reduceGradient(t1_grad, originalT1.shape, t1_.shape);
        originalT1.grad = originalT1.grad.map((g, i) => g + (t1_reduced_grad[i] || 0));

        const t2_grad = out.grad.map((og, i) => og * t1_.data[i]);
        const t2_reduced_grad = reduceGradient(t2_grad, originalT2.shape, t2_.shape);
        originalT2.grad = originalT2.grad.map((g, i) => g + (t2_reduced_grad[i] || 0));
    };
    return out;
}

export function dot(t1: Val, t2: Val) : Val {
    assert((t1.dim === 1 || t1.dim === 2) && (t2.dim === 1 || t2.dim === 2), () => `In dot: Both inputs must be dim 1 or 2. Got dims ${t1.dim} and ${t2.dim}`);
    if (t1.dim === 1 && t2.dim === 1) {
        return sum(mul(t1, t2))
    }

    const t1_ = t1.dim === 1 ? t1.reshape([1, t1.shape[0]]) : t1;
    const t2_ = t2.dim === 1 ? t2.reshape([t2.shape[0], 1]) : t2;

    assert(t1_.shape[1] === t2_.shape[0], ()=> `In dot: inner dimensions didn't match. dimensioins got: ${t1.shape} and ${t2.shape}`)

    const out_shape = [t1_.shape[0], t2_.shape[1]];
    const out_size = out_shape[0] * out_shape[1];
    const res_data = new Float64Array(out_size);

    for (let i=0; i<t1_.shape[0]; i++) {
        for (let j=0; j<t2_.shape[1]; j++) {
            let sum=0;
            for(let k=0; k<t1_.shape[1]; k++) {
                sum += t1_.data[i*t1_.shape[1] + k] * t2_.data[k*t2_.shape[1] + j]
            }
            res_data[i * out_shape[1] + j] = sum;
        }
    }

    const out = new Val(out_shape);
    out.data = res_data;
    out._prev = new Set([t1, t2]);

    out._backward = () => {
        // dL/dt1 = dL/dout . T(t2_)
        const gradT1_ = dot(out.gradVal(), t2_.T);
        const grad_to_accum_t1 = t1.dim === 1 ? gradT1_.reshape(t1.shape) : gradT1_;
        t1.grad = t1.grad.map((g, i) => g + grad_to_accum_t1.data[i]);
        
        // dL/dt2 = T(t1_) . dL/dout
        const gradT2_ = dot(t1_.T, out.gradVal());
        const grad_to_accum_t2 = t2.dim === 1 ? gradT2_.reshape(t2.shape) : gradT2_;
        t2.grad = t2.grad.map((g, i) => g + grad_to_accum_t2.data[i]);
    }

    if (t1.dim === 1 && t2.dim === 1) { // if both og inputs were 1D, out should also be scalar
        return out.reshape([1]);
    }
    return out;
}

export function pow(t: Val, num: number) : Val {
    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => k ** num)
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g + (num * t.data[i]**(num-1)) * out.grad[i]);
    }
    return out;
}

export function div(t: Val, num: number) : Val {
    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => k / num)
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g + (out.grad[i] / num));
    }
    return out;
}

export function divElementWise(t1: Val, t2: Val) : Val {
    assert(t1.dim === t2.dim, ()=> `In element wise division: Both matrices must have the same dim. got t1dim: ${t1.dim} and t2dim: ${t2.dim}`)
    assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In divElementWise: Both matrices must have the same shape')
    assert(t2.data.every((k:number)=> k !== 0), ()=> "Division by zero error in element-wise division");

    let out = new Val(t1.shape)
    out.data = t1.data.map((num: number, idx: number)=>num / t2.data[idx])
    out._prev = new Set([t1, t2])
    out._backward = () => {
        // dL/dt1 = dL/dout * d(t1/t2)/dt1 = dL/dout * (1/t2)
        t1.grad = t1.grad.map((g, i) => g + (1/t2.data[i]) * out.grad[i]);
        // dL/dt2 = dL/dout * d(t1/t2)/dt2 = dL/dout * (-t1 / t2^2)
        t2.grad = t2.grad.map((g, i) => g + (-t1.data[i]/t2.data[i]**2) * out.grad[i]);
    }
    return out;
}

export function negate(t: Val) : Val {
    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => -k)
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g - out.grad[i]);
    }
    return out;
}

export function abs(t: Val) : Val {
    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => Math.abs(k))
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g + (t.data[i] > 0 ? 1 : -1) * out.grad[i]);
    }
    return out;
}

export function exp(t: Val) : Val {
    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => Math.exp(k))
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g + out.data[i] * out.grad[i]);
    }
    return out;
}

export function log(t: Val) : Val {
    assert(t.data.every((k:number)=>k>0), ()=> "Log input must be positive")

    let out = new Val(t.shape)
    out.data = t.data.map((k:number) => Math.log(k))
    out._prev = new Set([t])
    out._backward = () => {
        t.grad = t.grad.map((g, i) => g + (1/t.data[i]) * out.grad[i]);
    }
    return out;
}

export function sum(t: Val, axis?: number, keepdims = false): Val {
    // Case 1: Sum all elements (no axis specified)
    if (axis === undefined) {
        const out = new Val(keepdims ? t.shape.map(() => 1) : [1]);
        out.data[0] = t.data.reduce((a: number, c: number) => a + c, 0);
        out._prev = new Set([t]);
        out._backward = () => {
            t.grad = t.grad.map(g => g + out.grad[0]);
        };
        return out;
    }

    // Case 2: Sum along a specific axis
    const new_shape = t.shape.map((dim, i) => 
        (i === axis && !keepdims) ? 1 : (i === axis ? 1 : dim)
    ).filter(dim => dim !== 1 || keepdims);

    const out = new Val(new_shape);
    const stride = t.shape.slice(axis + 1).reduce((a, b) => a * b, 1);

    // Forward pass (compute sums)
    for (let i = 0; i < out.size; i++) {
        let sum = 0;
        for (let j = 0; j < t.shape[axis]; j++) {
            const idx = Math.floor(i / stride) * t.shape[axis] * stride + 
                        j * stride + 
                        i % stride;
            sum += t.data[idx];
        }
        out.data[i] = sum;
    }

    // Backward
    out._prev = new Set([t]);
    out._backward = () => {
        for (let i = 0; i < out.size; i++) {
            for (let j = 0; j < t.shape[axis]; j++) {
                const idx = Math.floor(i / stride) * t.shape[axis] * stride + 
                            j * stride + 
                            i % stride;
                t.grad[idx] += out.grad[i];
            }
        }
    };

    return out;
}

export function mean(t: Val, axis?: number, keepdims = false) : Val {
    const N = axis === undefined ? t.size : t.shape[axis];
    const sum_val = sum(t, axis, keepdims);

    const out = div(sum_val, N);
    out._prev = new Set([t]);

    return out;
}

/**
 * X: input     : [batch_size, n_height, n_width, n_channels]
 * F: filter    : [c_outputchannels, filter_size, filter_size, n_channels]
 * st: stride   : default = 1
 * pad: padding : default = 0
 */
export function conv2d(X: Val, F: Val, st: number=1, pad: number=0) {
    assert(X.dim === 4, () => `conv2d: inuput x must be 4d`)
    assert(F.dim === 4, () => `conv2d: filter f must be 4d`)
    assert(F.shape[1] === F.shape[2], () => `conv2d: kernels must be square`)
    assert(X.shape[3] === F.shape[3], () => `conv2d: Input channels (${X.shape[3]}) must match kernel input channels (${F.shape[3]}`)
    assert(st > 0 && Number.isInteger(st), () => `conv2d: stride must be > 0`)
    assert(pad >= 0 && Number.isInteger(pad), () => `conv2d: padding must be >= 0`)

    const batch_size = X.shape[0];
    const H = X.shape[1];
    const W = X.shape[2];
    const C_IN = X.shape[3];
    const C_OUT = F.shape[0];
    const FS = F.shape[1];      // filter size

    // output dims
    const H_OUT = Math.floor((H - FS + 2*pad)/st) + 1;
    const W_OUT = Math.floor((W - FS + 2*pad)/st) + 1;

    if (H_OUT <= 0 || W_OUT <= 0) {
        throw new Error(`Conv2d: invalid output dims. Check input, filter, stride or padding`)
    }

    const outShape = [batch_size, H, W, C_OUT]
    const out = new Val(outShape);

    for (let batch=0; batch<batch_size; batch++) {              // iterating over batch
        for (let h_=0; h_<H_OUT; h_++) {                         // iterating over output height
            for (let w_=0; w_<W_OUT; w_++) {                     // iterating over output width
                for (let c_out=0; c_out<C_OUT; c_out++) {       // iterating over output channels (each kernel)

                    // input window start coordinates
                    const h_start = h_ * st - pad;
                    const w_start = w_ * st - pad;

                    let sum=0.0;

                    for (let f=0; f<FS*FS; f++) {
                        const fh = Math.floor(f/FS);            // filter row
                        const fw = f%FS;                        // filter col

                        for (let c_in=0; c_in<C_IN; c_in++) {   // iterating over filter channels
                            const h_in_idx = h_start + fh;
                            const w_in_idx = w_start + fw;

                            if (h_in_idx>=0 && h_in_idx<H && w_in_idx>=0 && w_in_idx<W) {
                                const x_idx = batch * (H*W*C_IN) + h_in_idx * (W*C_IN) + w_in_idx * C_IN + c_in;
                                const w_idx = c_out * (FS*FS*C_IN) + fh * (FS*C_IN) + fw * C_IN + c_in;

                                sum += X.data[x_idx] * F.data[w_idx];
                            }
                        }
                    }

                    const out_idx = batch * (H_OUT*W_OUT*C_OUT) + h_ * (W_OUT*C_OUT) + w_ * C_OUT + c_out;
                    out.data[out_idx] = sum;

                }
            }
        }
    }

    out._prev = new Set([X, F]);
    out._backward = () => {
        const dL_dOUT = out.grad;

        if (!X.grad || X.grad.length !== X.size) {
            console.warn(`conv2d backward: init grad for input X (shape ${X.shape})`)
            X.grad = new Float64Array(X.size).fill(0);
        }
        if (!F.grad || F.grad.length !== F.size) {
            console.warn(`conv2d backward: init grad for weights F(shape ${F.shape})`)
            F.grad = new Float64Array(F.size).fill(0);
        }

        for (let batch=0; batch<batch_size; batch++) {              // iterating over batch
            for (let h_=0; h_<H_OUT; h_++) {                         // iterating over output height
                for (let w_=0; w_<W_OUT; w_++) {                     // iterating over output width
                    for (let c_out=0; c_out<C_OUT; c_out++) {       // iterating over output channels (each kernel)
    
                        const out_grad_idx = batch * (H_OUT*W_OUT*C_OUT) + h_*(W_OUT*C_OUT) + w_*C_OUT + c_out;
                        const grad_val = dL_dOUT[out_grad_idx];

                        if (grad_val === 0) continue;

                        const h_start = h_ * st - pad;
                        const w_start = w_ * st - pad;
    
                        for (let f=0; f<FS*FS; f++) {
                            const fh = Math.floor(f/FS);            // filter row
                            const fw = f%FS;                        // filter col
    
                            for (let c_in=0; c_in<C_IN; c_in++) {   // iterating over filter channels
                                const h_in_idx = h_start + fh;
                                const w_in_idx = w_start + fw;
    
                                if (h_in_idx>=0 && h_in_idx<H && w_in_idx>=0 && w_in_idx<W) {
                                    const x_idx = batch * (H*W*C_IN)
                                                + h_in_idx * (W*C_IN)
                                                + w_in_idx * C_IN
                                                + c_in;
                                    
                                    const w_idx = c_out * (FS*FS*C_IN)
                                                + fh * (FS*C_IN)
                                                + fw * C_IN
                                                + c_in;
                                    
                                    // dL/dW += dL/dOut * dOut/dW = dL/dOut * X
                                    if (w_idx < F.grad.length) {
                                        F.grad[w_idx] += X.data[x_idx] * grad_val;
                                    }

                                    // dL/dX += dL/dOut * dOut/dX = dL/dOut * W
                                    if (x_idx < X.grad.length) {
                                        X.grad[x_idx] += F.data[w_idx] * grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}