import { NDArray, t_any } from './types'
import { Tensor } from './tensor';

export function assert(expr: boolean, msg: () => string) {
    if (!expr) {
      throw new Error(typeof msg === 'string' ? msg : msg());
    }
}

export function addNDarrays(arr1: NDArray, arr2: NDArray): NDArray {
    if (Array.isArray(arr1) && Array.isArray(arr2)) {
        return arr1.map((val, idx) => this.addNDarrays(val, arr2[idx]));
    } else {
        // Base case: both are numbers, so return their sum
        return (<number>arr1) + (<number>arr2);
    }
}

export function hadamardNDarrays(arr1: NDArray, arr2: NDArray): NDArray {
    if (Array.isArray(arr1) && Array.isArray(arr2)) {
        return arr1.map((val, idx) => this.hadamardNDarrays(val, arr2[idx]));
    } else {
        return (<number>arr1) * (<number>arr2);
    }
}

// This is a recursive function which will take the operation type (like * for multiply, ** for power, exp for exponent etc)
// and will return with the operation applied on the input ND array.
export function ophelper_(t: NDArray, op_type: string, num?: number): NDArray {
    if(Array.isArray(t)) {
        return t.map((val)=> ophelper_(val, op_type, num));
    }else {
        if (op_type === '**') {
            return (t ** num);
        } else if (op_type === 'exp') {
            return (Math.exp(t));
        } else if (op_type === 'tanh') {
            return ((Math.exp(2*t) - 1)/(Math.exp(2*t) + 1));
        } else if (op_type === '/') {
            return (t / num);
        }
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

export function createArray(shape: number[], value: number, randomfn?: () => number) : NDArray {
    // Recursively create an array with the given shape and fill with the given value
    if (shape.length === 0) return (randomfn? randomfn() : value) as any;

    const [head, ...tail] = shape;
    const arr: any[] = [];
    for (let i=0; i<head; i++) {
        arr.push(this.createArray(tail as any, value, randomfn));
    }
    return arr;
}