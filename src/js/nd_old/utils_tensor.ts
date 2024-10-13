import { Tensor } from "../nd_old/tensor";
import { t_any } from "../nd_old/types";
import { createArray } from "../nd_old/utils_nd";

export function convertToTensor(t: t_any) : Tensor {
    if (t instanceof Tensor) {
        return t
    } else if (typeof t === 'number' || Array.isArray(t)) {
        return new Tensor(t, '')
    } else {
        throw new Error("Unsupported input type for convertToTensor");
    }
}

export function broadcastAndConvertNum(t1: t_any, t2: t_any) : [Tensor, Tensor] {
    //rank check to make sure we're only broadcasting when the other tensor is not a scalar tensor aswell
    if (typeof t1 === 'number' && t2 instanceof Tensor && t2.rank !== 0) {
        t1 = convertToTensor(t1)
        t1.data = createArray(t2.shape, t1.data[0])
        t1.shape = t2.shape
    } else if (typeof t2 === 'number' && t1 instanceof Tensor && t1.rank !== 0) {
        t2 = convertToTensor(t2)
        t2.data = createArray(t1.shape, t2.data[0])
        t2.shape = t1.shape
    }
    t1 = convertToTensor(t1)
    t2 = convertToTensor(t2)
    return [t1, t2]
}