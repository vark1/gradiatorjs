import { NDArray, t_any } from './types'
import { Tensor } from './tensor';
import { add, mul, sub } from './ops';

export function assert(expr: boolean, msg: () => string) {
    if (!expr) {
      throw new Error(typeof msg === 'string' ? msg : msg());
    }
}

// Apply a function element-wise on a single ND array (Unary operation)
export function applyFnUnary(t: NDArray, fn: (x: number) => number): NDArray {
    if (Array.isArray(t)) {
        return t.map((val) => applyFnUnary(val, fn));
    } else {
        return fn(t);
    }
}

// Apply a function element-wise between two ND Arrays (Binary operation)
export function applyFnBinary(t1: NDArray, t2: NDArray, fn: (x: number, y: number) => number) : NDArray {
    if (Array.isArray(t1) && Array.isArray(t2)) {
        return t1.map((val, idx) => applyFnBinary(val, t2[idx], fn));
    } else {
        return fn(<number>t1, <number>t2);
    }
}

export function convertToTensor(t: t_any) : Tensor {
    if (t instanceof Tensor) {
        return t
    } else if (typeof t === 'number' || Array.isArray(t)) {
        return new Tensor(t, '')
    } else {
        throw new Error("Unsupported input type for convertToTensor");
    }
}

export function broadcastAndConvertNum(t1: t_any, t2: t_any) : [Tensor, Tensor] {
    //rank check to make sure we're only broadcasting when the other tensor is not a scalar tensor aswell
    if (typeof t1 === 'number' && t2 instanceof Tensor && t2.rank !== 0) {
        t1 = convertToTensor(t1)
        t1.data = createArray(t2.shape, t1.data[0])
        t1.shape = t2.shape
    } else if (typeof t2 === 'number' && t1 instanceof Tensor && t1.rank !== 0) {
        t2 = convertToTensor(t2)
        t2.data = createArray(t1.shape, t2.data[0])
        t2.shape = t1.shape
    }
    t1 = convertToTensor(t1)
    t2 = convertToTensor(t2)
    return [t1, t2]
}

// Recursively create an array with the given shape and fill with the given value
export function createArray(shape: number[], value: number, randomfn?: () => number) : NDArray {
    if (shape.length === 0) return (randomfn? randomfn() : value) as any;

    const [head, ...tail] = shape;
    const arr: any[] = [];
    for (let i=0; i<head; i++) {
        arr.push(this.createArray(tail as any, value, randomfn));
    }
    return arr;
}