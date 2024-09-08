import { Tensor } from './tensor';
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