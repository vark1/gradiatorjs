import { Val } from "./val.js";
import { MinMaxInfo } from "./js/types_and_interfaces/general.js";

export function assert(expr: boolean, msg: () => string) {
    if (!expr) {
        throw new Error(typeof msg === "string" ? msg : msg());
    }
}

export function gaussianRandom(mean=0, stdev=1) : number {
    const u = 1 - Math.random(); // Converting [0,1) to (0,1]
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stdev + mean;
}

export function arraysEqual(a: Float64Array, b: Float64Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

export function broadcast(t1: Val | number, t2: Val | number): [Val, Val] {
    const v1 = t1 instanceof Val ? t1 : new Val([], t1 as number);
    const v2 = t2 instanceof Val ? t2 : new Val([], t2 as number);

    // if shapes already match, return clones
    if (
        v1.shape.length === v2.shape.length &&
        v1.shape.every((dim, i) => dim === v2.shape[i])
    ) {
        return [v1.clone(), v2.clone()];
    }

    let v1_out = v1.clone();
    let v2_out = v2.clone();
    let broadcast_occurred = false;

    // scalars
    if (v1.size === 1 && v2.size > 1) {
        const shape = v2.shape;
        const data = new Float64Array(v2.size).fill(v1.data[0]);
        v1_out = new Val(shape); 
        v1_out.data = data;
        broadcast_occurred = true;

    } else if (v2.size === 1 && v1.size > 1) {
        const shape = v1.shape;
        const data = new Float64Array(v1.size).fill(v2.data[0]);
        v2_out = new Val(shape); 
        v2_out.data = data;
        broadcast_occurred = true;

    } else if (v1.size === 1 && v2.size === 1) {
        if (v1.shape.length === 0 && v2.shape.length > 0) {
            v2_out = new Val([], v2.data[0]);
            broadcast_occurred = true;
        } else if (v2.shape.length === 0 && v1.shape.length > 0) {
            v1_out = new Val([], v1.data[0]);
            broadcast_occurred = true;
        }
    }

    // limited 2D broadcasting
    else if (!broadcast_occurred && v1.dim === 2 && v2.dim === 2) {
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
        else if (v1.shape[0] === v2.shape[0] && v2.shape[1] === 1 && v1.shape[1] > 1) {
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
        else if (v1.shape[1] === v2.shape[1] && v1.shape[0] === 1 && v2.shape[0] > 1) {
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
        else if (v1.shape[1] === v2.shape[1] && v2.shape[0] === 1 && v1.shape[0] > 1) {
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
            v1_out = new Val(target_shape);
            v1_out.data = broadcasted_data;
            broadcast_occurred = true;
        } else if (needs_v2_broadcast && broadcasted_data) {
            v2_out = new Val(target_shape);
            v2_out.data = broadcasted_data;
            broadcast_occurred = true;
        }
    }

    if (v1.size > 1 && v2.size > 1 && !v1_out.shape.every((dim, i) => dim === v2_out.shape[i])) {
        assert(false, () => `Tensors could not be broadcast to compatible shapes. Original Shapes: ${v1.shape} and ${v2.shape}`);
    }

    return [v1_out, v2_out];
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
    if (originalShape.length === 0 || originalShape.length === 1) {
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

export function calculateMinMax(data: Float64Array): MinMaxInfo {
    if (!data || data.length === 0) {
        return { minv: 0, maxv: 0, dv: 0 };
    }
    let minv = data[0];
    let maxv = data[0];
    for (let i = 1; i < data.length; i++) {
        if (data[i] < minv) minv = data[i];
        if (data[i] > maxv) maxv = data[i];
    }
    return { minv, maxv, dv: maxv - minv };
}