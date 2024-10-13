import { assert } from "../utils/utils";

/* Everything in the neural net will be made of Val. */
export class Val{
    private _data: Float64Array;
    shape: number[];
    size: number;

    constructor(shape:number[], value?:number){
        this.shape = shape
        this.size = this.calculateSizeFromShape(this.shape)
        this._data = value? this.filled(this.size, value) : this.zeros(this.size)
    }

    set data(a: any) {
        if(a instanceof Float64Array){
            this._data = a
            return
        }
        let calcShape = this.calculateShape(a)
        let origShape = this.shape
        assert(
            typeof origShape !== 'undefined' 
            && origShape.length !== 0 
            && (calcShape.length === origShape.length && calcShape.every((v, i)=> v === origShape[i])), 
            ()=>`Shape of the matrix doesn't match the shape of Val. Shape of the matrix: ${calcShape} Shape of val: ${this.shape}`)    
        this._data = this.createArr(a)
    }

    private createArr(a) {
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

    private flattenND(a: any) {
        if (typeof a === 'number') {
            return [a]
        }
        let res = []
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
        // Calculate total number of elements in the tensor
        let size = shape[0];
        for (let i=1; i<shape.length; i++) {
            size *= shape[i];
        }
        return size;
    }

    private calculateShape(x): number[] {
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
        let x = new Val([...this.shape])
        x.data = this.data
        x.size = this.size
        return x
    }
}
