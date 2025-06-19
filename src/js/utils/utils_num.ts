import { Val } from '../Val/val.js';
import * as op from '../Val/ops.js'

export function gaussianRandom(mean=0, stdev=1) : number {
    const u = 1 - Math.random(); // Converting [0,1) to (0,1]
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stdev + mean;
}

export function meanSquaredErrorLoss(y_pred: Val, y_true: Val): Val {
    return op.div(op.sum(op.pow(op.sub(y_pred, y_true), 2)), y_true.size);
}

export function crossEntropyLoss_binary(y_pred: Val, y_true: Val, e: number = 1e-9): Val {
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

export function crossEntropyLoss_categorical(y_pred: Val, y_true: Val): Val {
    if (y_pred.shape.join(',') !== y_true.shape.join(',')) {
        throw new Error(`Shape mismatch for Cross-Entropy Loss. Pred: [${y_pred.shape}], True: [${y_true.shape}]`);
    }
    const epsilon = 1e-9;
    const log_pred = op.log(op.add(y_pred, epsilon));
    const product = op.mul(y_true, log_pred);
    const negatedSum = op.mul(op.sum(product,1), -1);
    return op.mean(negatedSum);
}

/**
 * A combined softmax and categorical cross-entropy loss function. This is done to avoid passing the 
 * backprop through a log fn. if the softmax outputs a probability P that is extremely close to zero, 
 * log(P) will be -infinity and grad (1/x) will explode. 
 * Backprop (dL/dZ) here will be just: predicted_probabilities - true_labels
 * @param logits raw, unnormalized output scores from the final dense layer (this is before softmax activation). [batch, numClasses]
 * @param y_true true labels, one-hot encoded. [batch, numClasses]
 */
export function crossEntropyLoss_softmax(logits: Val, y_true: Val): Val {
    if(logits.dim !== 2 || y_true.dim !== 2 || logits.shape.join(',') !== y_true.shape.join(',')) {
        throw new Error(`Shape mismatch for softmax `)
    }

    const batchSize = logits.shape[0];
    const numClasses = logits.shape[1];

    // applying softmax to get probs (P)
    const probs = new Val([batchSize, numClasses]);
    for (let b=0; b<batchSize; b++) {
        const rawOffset = b*numClasses;
        let max_logit = -Infinity;
        for (let j=0; j<numClasses; j++) {
            if (logits.data[rawOffset+j]>max_logit) {
                max_logit = logits.data[rawOffset+j];
            }
        }

        let sum_exps = 0;
        for (let j=0; j<numClasses; j++) {
            const exp_val = Math.exp(logits.data[rawOffset+j]-max_logit);
            probs.data[rawOffset+j] = exp_val;
            sum_exps += exp_val;
        }

        for (let j=0; j<numClasses; j++) {
            probs.data[rawOffset+j] /= sum_exps;
        }
    }

    // cross-entropy here
    const epsilon = 1e-9;
    let totalLoss = 0;
    for (let i=0; i<y_true.data.length; i++) {
        if (y_true.data[i]===1.0) {
            totalLoss += -Math.log(probs.data[i]+epsilon);
        }
    }
    const avgLoss = totalLoss/batchSize;
    const lossVal = new Val([], avgLoss);

    lossVal._prev = new Set([logits, y_true]);
    lossVal._backward = ()=> {
        const dL_dLogits = new Float64Array(logits.size);
        for (let i=0; i<logits.size; i++) {
            dL_dLogits[i] = (probs.data[i] - y_true.data[i])/batchSize;
        }

        if (!logits.grad || logits.grad.length !== logits.size) {
            logits.grad = new Float64Array(logits.size).fill(0);
        }
        for (let i=0; i<logits.size; i++) {
            logits.grad[i] += dL_dLogits[i];
        }
    };
    return lossVal;
}