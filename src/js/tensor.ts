import {NDArray} from './types'
import {createArray} from './utils/utils_nd'

/*
Tensor is the basic building block of all data in the neural net. You can 
use it to create scalar values, or use it to create 3D volumes of numbers.
*/
export class Tensor{
    data: NDArray;
    label: string;
    shape: number[];
    size: number;
    grad?: NDArray;
    op?: string;
    _prev: Tensor[] = [];
    _backward: Function

    constructor(data: NDArray, label: string, shape?:number[], op: string = "", prev: Tensor[] = []){
        this.data = data
        this.label = label

        let calcShape = this.calculateShape();
        if (typeof shape !== 'undefined' && shape.length !== 0 && !(calcShape.length === shape.length && calcShape.every(function(value, index) { return value === shape[index]}))) {
            throw new Error(`Shape provided doesn't match the shape of the Tensor provided. Shape of the tensor: ${calcShape} Shape provided: ${shape}`)
        }
        this.shape = calcShape
        this.size = this.calculateSizeFromShape(this.shape)
        this.grad = this.initializeGrad(this.shape)
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
    
    private initializeGrad(shape: number[]) : NDArray {
        // Initialize gradients with zeros
        return createArray(shape, 0 as any)
    }

    // Function to calculate the shape of an NDarr
    private calculateShape(): number[] {
        const shape: number[] = [];
        let current: any = this.data
        
        while (Array.isArray(current)) {
            shape.push(current.length);
            current = current[0];
        }
        return shape;
    }
    
    get rank() : number {
        return this.shape.length
    }

    public backpropagation() {
        let topo: Tensor[] = []
        let visited = new Set<Tensor>()
        function build_topo(v : Tensor) {
            if (!visited.has(v)) {
                visited.add(v)
                for (let child of v._prev) {
                    build_topo(child)
                }
                topo.push(v)
            }
        }
        build_topo(this)
        this.grad = createArray(this.shape, 1.0 as any)
        for (let node of topo.toReversed()) {
            node._backward()
        }
    }

    public print() {
        console.log("Value(label=" + this.label + ", data="+ this.data +", grad= " + this.grad + ")")
    }
}