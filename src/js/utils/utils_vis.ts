import { VISActivationData, LayerType } from "./types_and_interfaces.js";
import { assert, calculateMinMax } from "./utils.js";

export function getLayerColor(type: LayerType): string {
    const colors = {
        dense: '#2196F3',
        conv: '#FF9800',
        flatten: '#8cf800',
        maxpool: '#b5d3f2'
    };
    return colors[type] || '#999';
}

/**
 * 
 * @param element The html element we'll add all the canvas activation drawings to
 * @param A The VISActivationData with NHWC format ([Batch, Height, Width, C_out])
 * @param scale Multiplier for visualizations
 */
export function drawActivations(element: HTMLElement, A: VISActivationData, scale: number) {    
    assert(A.shape.length === 4, ()=>`Val provided is not 4d`)

    let s = scale || 2;

    let H = A.shape[1]
    let W = A.shape[2]
    let C = A.shape[3]

    const mm = calculateMinMax(A.activationSample);

    if (H === 0 || W === 0 || C === 0) return;
    
    for (let i=0; i<C; i++) {
        const canv = document.createElement('canvas');
        canv.className = 'act-map';
        const scaledW = W*s;
        const scaledH = H*s;
        canv.width = W;
        canv.height = H;
        const ctx = canv.getContext('2d');
        if (!ctx) continue;
        const g = ctx.createImageData(scaledW, scaledH);

        for (let y=0; y<H; y++) {
            for (let x=0; x<W; x++) {
                // Data is in NHWC: flat idx will be = (b*H*W*C) + (h*W*C) + (w*C) + c
                const idx = (y*W*C)+(x*C)+i;
                const val = A.activationSample[idx];
                const dval = mm.dv === 0 ? 128 : Math.floor((val-mm.minv)/mm.dv*255);

                for (let dy=0; dy<s; dy++) {
                    for (let dx=0; dx<s; dx++) {
                        const pp = ((scaledW*(y*s+dy))+(x*s+dx))*4;
                        for (let j=0; j<3; j++) {
                            g.data[pp+i] = dval;
                        }
                        g.data[pp + 3] = 255; // Alpha
                    }
                }
            }
        }

        ctx.putImageData(g, 0, 0);
        element.appendChild(canv);
    }
}