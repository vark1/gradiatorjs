import {NDarr} from './types'
/*
Tensor is the basic building block of all data in the neural net. You can 
use it to create scalar values, or use it to create 3D volumes of numbers.
*/
export class Tensor<T extends number[]>{
    label: string;
    data: NDarr<T>;
    shape: T;   // number[]
    size: number;
    grad?: NDarr<T>;

    constructor(data: NDarr<T>, shape:T, label: string){
        this.data = data
        this.shape = shape
        this.label = label
        this.size = this.calculateSizeFromShape(shape)
        this.grad = this.initializeGrad(shape)
    }

    private calculateSizeFromShape(shape: T): number {
        // Calculate total number of elements in the tensor
        let size = shape[0];
        for (let i=1; i<shape.length; i++) {
            size *= shape[i];
        }
        return size;
    }
    
    private initializeGrad(shape: T) : NDarr<T> {
        // Initialize gradients with zeroes
        return this.createArray(shape, 0 as any)
    }

    private createArray(shape: T, value: number) : NDarr<T> {
        // Recursively create an array with the given shape and fill with the given value
        if (shape.length == 0) return value as any;

        const [head, ...tail] = shape;
        const arr: any[] = [];
        for (let i=0; i<head; i++) {
            arr.push(this.createArray(tail as any, value));
        }
        return arr as NDarr<T>;
    }
}
