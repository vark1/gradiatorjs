import * as util from './utils'

/*
Tensor is the basic building block of all data in the neural net. You can 
use it to create scalar values, or use it to create 3D volumes of numbers.
*/
class Tensor {
    label: string;
    data: object;
    shape: number[];
    size: number;
    initVal: number;

    constructor(data: object, shape:number[], label: string=""){
        this.data = data
        this.shape = shape
        this.label = label
        this.size = util.getSizeFromShape(shape)
    }
}
