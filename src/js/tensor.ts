import {NDarr} from './types'
/*
Tensor is the basic building block of all data in the neural net. You can 
use it to create scalar values, or use it to create 3D volumes of numbers.
*/
export class Tensor<T extends number[]>{
    label: string;
    data: NDarr<T>;
    shape: number[];
    size: number;
    grad?: NDarr<T>;
    op?: string;
    _prev: Tensor<T>[] = [];
    _backward: Function

    constructor(data: NDarr<T>, shape:number[], label: string, op: string = "", prev: Tensor<T>[] = []){
        this.data = data
        this.shape = shape
        this.label = label
        this.size = this.calculateSizeFromShape(shape)
        this.grad = this.initializeGrad(shape)
        this.op = op
        this._prev = prev
        this._backward = function(): void {}
    }

    private calculateSizeFromShape(shape: number[]): number {
        // Calculate total number of elements in the tensor
        let size = shape[0];
        for (let i=1; i<shape.length; i++) {
            size *= shape[i];
        }
        return size;
    }
    
    private initializeGrad(shape: number[]) : NDarr<T> {
        // Initialize gradients with zeroes
        return this.createArray(shape, 0 as any)
    }

    static zeroes<T extends number[]>(shape: number[], label: string = 'zeroes') : Tensor<T> {
        const data = new Tensor<T>([] as NDarr<T>, shape, label).createArray(shape, 0)
        return new Tensor(data, shape, label)
    }

    static random<T extends number[]>(shape: T, label: string = 'random') : Tensor<T> {
        const data = new Tensor<T>([] as NDarr<T>, shape, label).createArray(shape, 0, ()=> Math.random())
        return new Tensor(data, shape, label)
    }

    private createArray(shape: number[], value: number, randomfn?: () => number) : NDarr<T> {
        // Recursively create an array with the given shape and fill with the given value
        if (shape.length === 0) return (randomfn? randomfn() : value) as any;

        const [head, ...tail] = shape;
        const arr: any[] = [];
        for (let i=0; i<head; i++) {
            arr.push(this.createArray(tail as any, value, randomfn));
        }
        return arr as NDarr<T>;
    }

    get rank() : number {
        return this.shape.length
    }

    public backpropagation() {
        let topo: Tensor<T>[] = []
        let visited = new Set<Tensor<T>>()
        function build_topo(v : Tensor<T>) {
            if (!visited.has(v)) {
                visited.add(v)
                for (let child of v._prev) {
                    build_topo(child)
                }
                topo.push(v)
            }
        }
        build_topo(this)
        this.grad = this.createArray(this.shape, 1.0 as any)
        for (let node of topo.toReversed()) {
            node._backward()
        }
    }
}
