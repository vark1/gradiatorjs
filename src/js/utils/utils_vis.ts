import { LayerType, MinMaxInfo } from "../types_and_interfaces/general.js";
import { VISActivationData } from "../types_and_interfaces/vis_interfaces.js";
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

function findOrCreateCanvas(wrapper: HTMLElement, canvasId: string): HTMLCanvasElement {
    let canv = <HTMLCanvasElement>document.getElementById(canvasId);
    if (canv)   return canv;

    canv = document.createElement('canvas');
    canv.id = canvasId;
    canv.style.border = '1px solid #ccc';
    canv.style.backgroundColor = '#f8f8f8';
    canv.style.imageRendering = 'pixelated';
    wrapper.appendChild(canv);
    
    return canv;
}

export function drawActivations(canvWrapper: HTMLElement, actData: VISActivationData, vizLayerId: string, showActivations: boolean) {

    let shape = actData.zShape;
    let sample = actData.zSample?.data;
    if (showActivations) {
        shape = actData.aShape;
        sample = actData.aSample?.data;
    }

    if (shape.length !== 4 && shape[0] !== 1 || !sample || sample.length === 0) {
        console.log(`Conv shape error (${shape.join(',')})`);
        return;
    }

    const [_, H_out, W_out, C_out] = shape
    const singleMapSize = H_out * W_out;

    if (singleMapSize <= 0 || C_out <= 0 || sample.length !== singleMapSize * C_out) {
        console.log(`Conv data error (S:${sample.length} vs E:${singleMapSize * C_out})`);
        return;
    }

    const mm = calculateMinMax(sample);

    for (let c=0; c<C_out; c++) {
        const canvasId = `act-canvas-${vizLayerId}-map${c}`;
        const canv = findOrCreateCanvas(canvWrapper, canvasId);

        // xtracting data for the feature map (channel x)
        const xMapData = new Float64Array(singleMapSize);
        for (let i=0; i<singleMapSize; i++) {
            xMapData[i] = sample[i * C_out + c];
        }

        canv.width = W_out;
        canv.height = H_out;

        const maxDisplayDim = 48;
        let displayW, displayH;
        if (W_out >= H_out) { // wider or square
            displayW = maxDisplayDim;
            displayH = Math.round(maxDisplayDim * (H_out / W_out)) || 1;
        } else {              // taller
            displayH = maxDisplayDim;
            displayW = Math.round(maxDisplayDim * (W_out / H_out)) || 1;
        }
        canv.style.width = `${displayW}px`;
        canv.style.height = `${displayH}px`;
        canv.title = `Channel ${c + 1}/${C_out} (Size: ${W_out}x${H_out})`;

        // Rendering to canvas
        const ctx = canv.getContext('2d')
        if (!ctx) continue;

        ctx.clearRect(0, 0, canv.width, canv.height);
        const imageData = ctx.createImageData(W_out, H_out);
        const data = imageData.data;

        for (let i=0; i<xMapData.length; i++) {
            const normalizedValue = mm.dv === 0 ? 0.5 : (xMapData[i] - mm.minv) / mm.dv;
            const grayVal = Math.max(0, Math.min(255, Math.round(normalizedValue * 255)));
            const pixelIdx = i*4;
            data[pixelIdx] = data[pixelIdx+1] = data[pixelIdx+2] = grayVal;
            data[pixelIdx + 3] = 255;
        }
        ctx.imageSmoothingEnabled = false;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = W_out;
        tempCanvas.height = H_out;
        tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
        ctx.drawImage(tempCanvas, 0, 0, canv.width, canv.height);
    }

    // removing existing canvases incase someone changes the outchannels
    const existingCanvases = canvWrapper.querySelectorAll('canvas');
    if (existingCanvases.length > C_out) {
        for (let i = C_out; i < existingCanvases.length; i++) {
            existingCanvases[i].remove();
        }
    }
}

export function drawHeatMap1D(canvWrapper: HTMLElement, actData: VISActivationData, vizLayerId: string, showActivations: boolean) {

    let shape = actData.zShape;
    let sample = actData.zSample?.data;
    if (showActivations) {
        shape = actData.aShape;
        sample = actData.aSample?.data;
    }

    if(shape.length !== 2 || shape[0] !== 1) {
        console.log(`Dense shape error (${shape.join(',')})`);
        return;
    }
    if (!sample || sample.length === 0) {
        console.log("no activation data");
        return;
    }

    const numActivations = shape[1];
    const canvasId = `act-canvas-${vizLayerId}-heatmap`;
    const canv = findOrCreateCanvas(canvWrapper, canvasId)
    canv.style.width = '100%';
    canv.style.height = '15px';
    canv.width = Math.min(numActivations, 256);
    canv.height = 15;

    const mm = calculateMinMax(sample)

    // rendering
    const ctx = canv.getContext('2d')
    if (!ctx) return;

    ctx.clearRect(0, 0, canv.width, canv.height);

    const scale = mm.dv === 0? 1: 255/mm.dv;

    const blockWidth = canv.width / numActivations;
    for (let i=0; i<numActivations && i<sample.length; i++) {
        const normalizedVal = mm.dv === 0 ? 0.5 : (sample[i] - mm.minv)/mm.dv;
        const grayVal = Math.max(0, Math.min(255, Math.round(normalizedVal * 255)))
        ctx.fillStyle = `rgb(${grayVal}, ${grayVal}, ${grayVal})`;
        ctx.fillRect(Math.floor(i*blockWidth), 0, Math.ceil(blockWidth), canv.height);
    }
}