import { Val } from "Val/val";

export function assert(expr: boolean, msg: () => string) {
    if (!expr) {
        throw new Error(typeof msg === "string" ? msg : msg());
    }
}

export function arraysEqual(a: Float64Array, b: Float64Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

export function broadcast(t1: Val | number, t2: Val | number): [Val, Val] {
    let v1 = t1 instanceof Val ? t1 : new Val([], t1 as number);
    let v2 = t2 instanceof Val ? t2 : new Val([], t2 as number);
    
    // if shapes already match, return clones
    if (
        v1.shape.length === v2.shape.length &&
        v1.shape.every((dim, i) => dim === v2.shape[i])
    ) {
        return [v1, v2];
    }

    // scalars
    if (v1.size === 1 && v2.size > 1) {
        const shape = v2.shape;
        const data = new Float64Array(v2.size).fill(v1.data[0]);
        v1 = new Val(shape);
        v1.data = data;
        return [v1, v2];
    }
    if (v2.size === 1 && v1.size > 1) {
        const shape = v1.shape;
        const data = new Float64Array(v1.size).fill(v2.data[0]);
        v2 = new Val(shape);
        v2.data = data;
        return [v1, v2];
    }

    // limited 2D broadcasting
    if (v1.dim === 2 && v2.dim === 2) {
        let broadcasted_data: Float64Array | null = null;
        let target_shape: number[] = [];
        let needs_v1_broadcast = false;
        let needs_v2_broadcast = false;

        // [M, 1] and [M, N] -> broadcast v1 to [M, N]
        if (v1.shape[0] === v2.shape[0] && v1.shape[1] === 1 && v2.shape[1] > 1) {
            target_shape = v2.shape;
            broadcasted_data = new Float64Array(v2.size);
            for (let r = 0; r < target_shape[0]; r++) {
                for (let c = 0; c < target_shape[1]; c++) {
                    broadcasted_data[r * target_shape[1] + c] = v1.data[r];
                }
            }
            needs_v1_broadcast = true;
        }

        // [M, N] and [M, 1] -> broadcast v2 to [M, N]
        else if (
            v1.shape[0] === v2.shape[0] &&
            v2.shape[1] === 1 &&
            v1.shape[1] > 1
        ) {
            target_shape = v1.shape;
            broadcasted_data = new Float64Array(v1.size);
            for (let r = 0; r < target_shape[0]; r++) {
                for (let c = 0; c < target_shape[1]; c++) {
                    broadcasted_data[r * target_shape[1] + c] = v2.data[r];
                }
            }
            needs_v2_broadcast = true;
        }

        // [1, N] and [M, N] -> broadcast v1 to [M, N]
        else if (
            v1.shape[1] === v2.shape[1] &&
            v1.shape[0] === 1 &&
            v2.shape[0] > 1
        ) {
            target_shape = v2.shape;
            broadcasted_data = new Float64Array(v2.size);
            for (let r = 0; r < target_shape[0]; r++) {
                for (let c = 0; c < target_shape[1]; c++) {
                    broadcasted_data[r * target_shape[1] + c] = v1.data[c];
                }
            }
            needs_v1_broadcast = true;
        }
        
        // [M, N] and [1, N] -> broadcast v2 to [M, N]
        else if (
            v1.shape[1] === v2.shape[1] &&
            v2.shape[0] === 1 &&
            v1.shape[0] > 1
        ) {
            target_shape = v1.shape;
            broadcasted_data = new Float64Array(v1.size);
            for (let r = 0; r < target_shape[0]; r++) {
                for (let c = 0; c < target_shape[1]; c++) {
                    broadcasted_data[r * target_shape[1] + c] = v2.data[c];
                }
            }
            needs_v2_broadcast = true;
        }

        if (needs_v1_broadcast && broadcasted_data) {
            v1 = new Val(target_shape);
            v1.data = broadcasted_data;
        } else if (needs_v2_broadcast && broadcasted_data) {
            v2 = new Val(target_shape);
            v2.data = broadcasted_data;
        }
    }

    assert(
        v1.shape.length === v2.shape.length &&
        v1.shape.every((dim, i) => dim === v2.shape[i]),
        () => `Tensors could not be broadcast to compatible shapes. Shapes: ${t1 instanceof Val ? t1.shape : "[scalar]"} and ${t2 instanceof Val ? t2.shape : "[scalar]"}`
    );

    return [v1, v2];
}

// This reduces a gradient calculated for a broadcasted shape back to the original shape (this is to preserve the gradients)
export function reduceGradient(
    gradient: Float64Array,
    originalShape: number[],
    broadcastedShape: number[]
): Float64Array {
    // Case: If No broadcasting occurred
    if (
        originalShape.length === broadcastedShape.length &&
        originalShape.every((dim, i) => dim === broadcastedShape[i])
    ) {return gradient;}

    const originalSize = originalShape.reduce((a, b) => a * b, 1);

    // Case: original was scalar
    if (originalShape.length === 0) {
        let sum = 0;
        for (let i = 0; i < gradient.length; i++) {
            sum += gradient[i];
        }
        return new Float64Array([sum]);
    }

    const reducedGrad = new Float64Array(originalSize).fill(0);

    // Case: 2D Broadcasting Reductions
    // Check if dimensions match (should be 2D for this block)
    if (originalShape.length === 2 && broadcastedShape.length === 2) {
        const [origRows, origCols] = originalShape;
        const [bcastRows, bcastCols] = broadcastedShape;

        const reduceCols = origCols === 1 && bcastCols > 1; // sum along axis 1 ([M, 1] -> [M, N])
        const reduceRows = origRows === 1 && bcastRows > 1; // Sum along axis 0 ([1, N] -> [M, N])

        if (reduceCols && reduceRows) {
        // [1, 1] -> [M, N], treat as scalar reduction
        let sum = 0;
        for (let i = 0; i < gradient.length; i++) {
            sum += gradient[i];
        }
        if (reducedGrad.length === 1) 
            reducedGrad[0] = sum;
        } else if (reduceCols && !reduceRows) {
            // [M, 1] -> [M, N], sum along axis 1 (cols)
            for (let r = 0; r < bcastRows; r++) {
                let sum = 0;
                for (let c = 0; c < bcastCols; c++) {
                    sum += gradient[r * bcastCols + c];
                }
                if (r < reducedGrad.length) reducedGrad[r] = sum;
            }
        } else if (reduceRows && !reduceCols) {
            // [1, N] -> [M, N], sum along axis 0 (rows)
            for (let c = 0; c < bcastCols; c++) {
                let sum = 0;
                for (let r = 0; r < bcastRows; r++) {
                    sum += gradient[r * bcastCols + c];
                }
                if (c < reducedGrad.length) reducedGrad[c] = sum;
            }
        } else {
            // Invalid broadcast pair
            if (gradient.length === reducedGrad.length) return gradient;
            console.error(
                `reduceGradient: Unhandled 2D case from ${broadcastedShape} to ${originalShape}`
            );
        }
        return reducedGrad;
    }
    // This indicates either N-D broadcasting (needs specific reduction logic) or an invalid state.
    console.warn(
        `reduceGradient: Unhandled broadcast reduction from ${broadcastedShape} to ${originalShape}. Returning zero gradient.`
    );
    return reducedGrad; // Return the zero-filled array matching original size
}