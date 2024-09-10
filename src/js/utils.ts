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

// This is a recursive function which will take the operation type (like * for multiply, ** for power, exp for exponent etc)
// and will return with the operation applied on the input ND array.
export function ophelper_<T extends number[]>(t: NDarr<T>, op_type: string, num?: number): NDarr<T> {
    if(Array.isArray(t)) {
        return t.map((val)=> ophelper_(val as unknown as NDarr<T>, op_type, num)) as unknown as NDarr<T>;
    }else {
        if (op_type === '*') {
            return (t * num) as unknown as NDarr<T>;
        } else if (op_type === '**') {
            return (t ** num) as unknown as NDarr<T>;
        } else if (op_type === 'exp') {
            return (Math.exp(t)) as unknown as NDarr<T>;
        } else if (op_type === 'tanh') {
            return ((Math.exp(2*t) - 1)/(Math.exp(2*t) + 1)) as unknown as NDarr<T>;
        }
    }
}

// (1, x)*(x, 1) | rank 1 * rank 1 | number[]*number[] => (1, 1)
function matmul1(a1: number[], a2: number[]) : number[]{
    let sum = 0;
    for(let i=0; i<a1.length; i++) {
        sum+=a1[i]*a2[i]
    }
    return [sum]
}

// (1, x)*(x, y) => (1, y) | rank 1 * rank 2 = (1, y)
function matmul2(a1: number[], a2: number[][]) {

}