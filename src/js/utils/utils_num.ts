import { Val } from '../Val/val.js';
import * as op from '../Val/ops.js'
import { sigmoid } from '../Val/activations.js';

export function gaussianRandom(mean=0, stdev=1) : number {
    const u = 1 - Math.random(); // Converting [0,1) to (0,1]
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stdev + mean;
}

export function meanSquaredErrorLoss(y_pred: Val, y_true: Val): Val {
    return op.div(op.sum(op.pow(op.sub(y_pred, y_true), 2)), y_true.size);
}

export function crossEntropyLoss(y_pred: Val, y_true: Val, e: number = 1e-9): Val {
    if (y_pred.shape.join(',') !== y_true.shape.join(',')) {
        throw new Error(`Shape mismatch for BCE Loss: pred ${y_pred.shape}, true ${y_true.shape}`);
    }
    const batch_size = y_true.shape[0] || 1;

    const t1 = op.add(y_pred, e);
    const t2 = op.add(op.sub(1, y_pred), e);

    // cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
    let total_sum = op.sum(op.add(
        op.mul(y_true, op.log(t1)),
        op.mul(op.sub(1, y_true), op.log(t2))
    ));
    if (batch_size === 0) return new Val([], 0);
    let avg_loss = op.mul(-1/batch_size, total_sum)
    return avg_loss;
}