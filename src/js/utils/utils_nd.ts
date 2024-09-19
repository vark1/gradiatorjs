import { NDArray, t_any } from '../types'

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

// Recursively create an array with the given shape and fill with the given value
export function createArray(shape: number[], value: number, randomfn?: () => number) : NDArray {
    if (shape.length === 0) return (randomfn? randomfn() : value) as any;

    const [head, ...tail] = shape;
    const arr: any[] = [];
    for (let i=0; i<head; i++) {
        arr.push(createArray(tail as any, value, randomfn));
    }
    return arr;
}

export function ndzeros(shape: number[]) : NDArray {
    return createArray(shape, 0)
}

export function ndrandom(shape: number[]) : NDArray {
    return createArray(shape, 0, ()=> Math.random())
}

// TODO: add reshape and resize