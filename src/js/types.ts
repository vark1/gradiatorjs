import { Tensor } from "./tensor"

export type NDarr<T extends any[]> = T extends [any, any, ...any] ? (T extends [any, ...infer K] ? NDarr <K>[] : number[]) : number[]

export function createTensor<T extends number[]>(shape: [...T]): NDarr<T> {
    return [] as any
}

export type t_any<T extends number[]> = Tensor<T> | number | NDarr<T>