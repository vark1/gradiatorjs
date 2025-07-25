import { assert } from "./utils.js";
import { gaussianRandom } from "./utils.js";

/* Everything in the neural net will be made of Val. */
/*
Scalars are represented as : shape = [], size = 1

*/
export class Val{
    private _data: Float64Array;
    public grad: Float64Array;
    _backward: () => void;
    _prev: Set<Val>;
    shape: number[];
    size: number;

    constructor(shape:number[], value?:number){
        this.shape = shape;
        this.size = this.calculateSizeFromShape(this.shape);
        this._data = value? this.filled(this.size, value) : this.zeros(this.size);
        this.grad = new Float64Array(this.size);
        this._backward = ()=> {};
        this._prev = new Set();
    }

    backward() {
        this.grad.fill(0);
        if (this.size === 1) this.grad[0] = 1;

        // Topological sort
        const topo: Val[] = [];
        const visited = new Set<Val>();
        const buildTopo = (v: Val) => {
            if(!visited.has(v)) {
                visited.add(v);
                v._prev.forEach(buildTopo);
                topo.push(v);
            }
        };
        buildTopo(this);
        topo.reverse().forEach(v=>{
            if (v._backward) v._backward();
        });
    }

    set data(a: any) {
        if(a instanceof Float64Array){
            this._data = new Float64Array(a);
            return;
        }
        let calcShape = this.calculateShape(a)
        let origShape = this.shape
        assert(
            typeof origShape !== 'undefined' 
            && origShape.length !== 0 
            && (calcShape.length === origShape.length && calcShape.every((v, i)=> v === origShape[i])), 
            ()=>`Shape of the matrix doesn't match the shape of Val. Shape of the matrix: ${calcShape} Shape of val: ${this.shape}`)    
        this._data = this.createArr(a)
        this.grad = new Float64Array(this.size)
    }

    private createArr(a: any) {
        if (typeof a === "number") {
            let x = new Float64Array(1)
            x[0] = a
            return x
        } else if (Array.isArray(a)) {
            return Float64Array.from(this.flattenND(a))
        } else {
            return new Float64Array(0)
        }
    }

    private flattenND(a: any) : number[]{
        if (typeof a === 'number') {
            return [a]
        }
        let res : number[] = []
        if (Array.isArray(a)) {
            for (let e of a) {
                res = res.concat(this.flattenND(e))
            }
        }
        return res
    }

    get data() {
        return this._data
    }
    
    private calculateSizeFromShape(shape: number[]): number {
        if (shape.length === 0) return 1; 
        return shape.reduce((acc, dim) => acc * dim, 1)
    }

    private calculateShape(x: any): number[] {
        const shape: number[] = [];
        let current: any = x
        
        while (Array.isArray(current)) {
            shape.push(current.length);
            current = current[0];
        }
        return shape;
    }

    private zeros(size: number) {
        return new Float64Array(size)
    }
    
    get dim() {
        return this.shape.length
    }

    private filled(size: number, value: number) {
        let x = new Float64Array(size)
        x.fill(value)
        return x
    }

    clone() : Val {
        let x = new Val([...this.shape]);
        x._data = Float64Array.from(this._data);
        x._backward = this._backward;
        x._prev = new Set(this._prev);
        x.grad = Float64Array.from(this.grad);

        return x;
    }

    get T() : Val {
        // Note: Only supports 2d arrays (for now)
        if(this.dim === 1) return this.clone(); // returning clone here for mutation issues
        assert(this.dim === 2, () => 'transpose only supports 2D arrays');
        let newShape = [this.shape[1]!, this.shape[0]!];
        let res = new Val(newShape);
        let x = new Float64Array(this.size)
        let y = this.data
    
        for (let i=0; i<this.shape[0]!; i++) {
            for (let j=0; j<this.shape[1]!; j++) {
                x[j*this.shape[0]! + i] = y[i*this.shape[1]! + j]
            }
        }
        res._data = x;
        res._prev = new Set([this]);
        res._backward = () => {
            // Transpose incoming gradient (res.grad) and accumulate it to the original tensor's grad (this.grad)
            for(let i=0; i<this.shape[0]!; i++) {
                for(let j=0; j<this.shape[1]!; j++) {
                    this.grad[i*this.shape[1]! + j]! += res.grad[j*this.shape[0]! + i]!;
                }
            }
        };
        return res;
    }

    reshape(newShape: number[]): Val {
        const inferredShape = [...newShape];

        const requiredSize = inferredShape.reduce((a, b) => a * b, 1);
        assert(this.size == requiredSize, ()=>`Cannot reshape array: number of elements (${this.size}) does not match the required size (${requiredSize}) for shape ${inferredShape}`);

        let result = new Val(inferredShape);
        result._data = Float64Array.from(this.data);
        result._prev = new Set([this]);

        const inputVal = this;  // storing this cuz javascript :D

        result._backward = () => {
            if (!inputVal.grad || inputVal.grad.length !== inputVal.size) {
                console.warn(`Reshape backward: Initializing gradient for input tensor (shape ${inputVal.shape})`);
                inputVal.grad = new Float64Array(inputVal.size).fill(0);
            }

            if (!result.grad || result.grad.length !== result.size) {
                console.warn(`Reshape backward: Gradient for reshaped tensor (shape ${result.shape}) is missing or has wrong size (${result.grad?.length} vs ${result.size}). Skipping accumulation.`);
                return;
            }

            for (let i = 0; i < inputVal.grad.length; i++) {
                if (i < result.grad.length) {
                    inputVal.grad[i]! += result.grad[i]!; // Accumulate gradient onto the captured input Val's grad
                }
            }
        };

        return result
    }

    randn() : Val{
        let x = new Val(this.shape)
        for (let i=0; i<this.size; i++) {
            x.data[i] = gaussianRandom()
        }
        return x
    }

    gradVal(): Val {
        const x = new Val(this.shape);
        x.data = Float64Array.from(this.grad)
        return x;
    }
}