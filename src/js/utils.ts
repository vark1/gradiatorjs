import { NDarr, t_any } from './types'
import { Tensor} from './tensor';

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
        if (op_type === '**') {
            return (t ** num) as unknown as NDarr<T>;
        } else if (op_type === 'exp') {
            return (Math.exp(t)) as unknown as NDarr<T>;
        } else if (op_type === 'tanh') {
            return ((Math.exp(2*t) - 1)/(Math.exp(2*t) + 1)) as unknown as NDarr<T>;
        } else if (op_type === '/') {
            return (t / num) as unknown as NDarr<T>;
        }
    }
}

// Function to calculate the shape of an NDarr
export function calculateShape(arr: number[] | NDarr<any>): number[] {
    // Base case: if it's a flat array (not an array of arrays), return an empty shape
    if (!Array.isArray(arr) || arr.length === 0 || typeof arr[0] !== 'object') {
        return [];
    }

    // Recursive case: determine the shape of the nested array
    const shape: number[] = [];
    let currentLevel: any[] = arr;
    
    while (Array.isArray(currentLevel) && currentLevel.length > 0) {
        shape.push(currentLevel.length);
        currentLevel = currentLevel[0];
    }
    
    return shape;
}

export function convertToTensor<T extends number[]>(t: t_any<T>) : Tensor<T> {
    if (t instanceof Tensor) {
        return t
    } else if (typeof t === 'number') {
        return new Tensor([t] as NDarr<T>, [1], '')
    } else if (Array.isArray(t)) {
        return new Tensor(t, calculateShape(t), '')
    } else {
        throw new Error("Unsupported input type for convertToTensor");
    }
}

export function broadcastAndConvertNum<T extends number[]>(t1: t_any<T>, t2: t_any<T>) : [Tensor<T>, Tensor<T>] {
    if (typeof t1 === 'number' && t2 instanceof Tensor) {
        t1 = convertToTensor(t1)
        t1.data = createArray(t2.shape, t1.data[0])
        t1.shape = t2.shape
    } else if (typeof t2 === 'number' && t1 instanceof Tensor) {
        t2 = convertToTensor(t2)
        t2.data = createArray(t1.shape, t2.data[0])
        t2.shape = t1.shape
    }
    t1 = convertToTensor(t1)
    t2 = convertToTensor(t2)
    return [t1, t2]
}

export function createArray<T extends number[]>(shape: number[], value: number, randomfn?: () => number) : NDarr<T> {
    // Recursively create an array with the given shape and fill with the given value
    if (shape.length === 0) return (randomfn? randomfn() : value) as any;

    const [head, ...tail] = shape;
    const arr: any[] = [];
    for (let i=0; i<head; i++) {
        arr.push(this.createArray(tail as any, value, randomfn));
    }
    return arr as NDarr<T>;
}