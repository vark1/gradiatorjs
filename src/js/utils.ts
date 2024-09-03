export function getSizeFromShape(shape: number[]): number {
    if (shape.length == 0) {
        return 1;
    }
    let size = shape[0];
    for (let i=1; i<shape.length; i++) {
        size *= shape[i];
    }
    return size;
}