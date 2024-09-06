export type NDarr<T extends any[]> = T extends [any, any, ...any] ? (T extends [any, ...infer K] ? NDarr <K>[] : number[]) : number[]

export function createTensor<T extends number[]>(shape: [...T]): NDarr<T> {
    return [] as any
}
