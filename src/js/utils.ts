import { NDarr } from './types'

export function assert(expr: boolean, msg: () => string) {
    if (!expr) {
      throw new Error(typeof msg === 'string' ? msg : msg());
    }
}

export function addNDarrays<T extends number[]>(tensor1: NDarr<T>, tensor2: NDarr<T>): NDarr<T> {
    if (Array.isArray(tensor1) && Array.isArray(tensor2)) {
        return tensor1.map((val, idx) => this.addNDarrays(val, tensor2[idx])) as NDarr<T>;
    } else {
        // Base case: both are numbers, so return their sum
        // `<number><unknown>tensor1` is same as `tensor1 as unknown as number`
        return (<number><unknown>tensor1) + (<number><unknown>tensor2) as unknown as NDarr<T>;
    }
}

export function hadamardNDarrays<T extends number[]>(arr1: NDarr<T>, arr2: NDarr<T>): NDarr<T> {
    if (Array.isArray(arr1) && Array.isArray(arr2)) {
        return arr1.map((val, idx) => this.hadamardNDarrays(val, arr2[idx])) as NDarr<T>;
    } else {
        return (<number><unknown>arr1) * (<number><unknown>arr2) as unknown as NDarr<T>;
    }
}

// This is a recursive function which will take the operation type (like * for multiply, ** for power, exp for exponent etc)
// and will return with the operation applied on the input ND array.
export function ophelper_<T extends number[]>(t: NDarr<T>, op_type: string, num?: number): NDarr<T> {
    if(Array.isArray(t)) {
        return t.map((val)=> ophelper_(val as unknown as NDarr<T>, op_type, num)) as unknown as NDarr<T>;
    }else {
        if (op_type === '+') {
            return (t + num) as unknown as NDarr<T>;
        } else if (op_type === '*') {
            return (t * num) as unknown as NDarr<T>;
        } else if (op_type === '**') {
            return (t ** num) as unknown as NDarr<T>;
        } else if (op_type === '-') {
            return (t - num) as unknown as NDarr<T>;
        } else if (op_type === 'exp') {
            return (Math.exp(t)) as unknown as NDarr<T>;
        } else if (op_type === 'tanh') {
            return ((Math.exp(2*t) - 1)/(Math.exp(2*t) + 1)) as unknown as NDarr<T>;
        } else if (op_type === '/') {
            return (t / num) as unknown as NDarr<T>;
        }
    }
}