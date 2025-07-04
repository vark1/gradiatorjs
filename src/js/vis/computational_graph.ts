import { Conv, Dense, Flatten, MaxPool2D, Module, Sequential } from "../../layers.js";
import { calculateMinMax } from "../../utils.js";
import { Val } from "../../val.js";
import { LayerOutputData } from "../types_and_interfaces/vis_interfaces.js";

function renderFeatureMap(canvas: HTMLCanvasElement, mapData: Float64Array, W: number, H: number, C: number) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = W;
    canvas.height = H;

    const imageData = ctx.createImageData(W, H);
    const dataArr = imageData.data;

    if (C===1){ //grayscale (like MNIST)
        const mm = calculateMinMax(mapData);
        for (let i=0; i<mapData.length; i++) {
            const normalized = mm.dv===0?0.5:(mapData[i]-mm.minv)/mm.dv;
            const grayVal = Math.floor(normalized*255);
            const p = i*4;
            dataArr[p] = dataArr[p+1] = dataArr[p+2] = grayVal;
            dataArr[p+3] = 255;
        }
    } else if (C===3){  //RGB
        for (let i=0; i<W*H; i++) {
            const pout=i*4;
            const pin = i*3;
            dataArr[pout] = mapData[pin] * 255;
            dataArr[pout + 1] = mapData[pin + 1] * 255;
            dataArr[pout + 2] = mapData[pin + 2] * 255;
            dataArr[pout + 3] = 255;
        }
    }
    ctx.putImageData(imageData, 0, 0);
}

// fn to render single activation column (Z or A)
function renderActivationCol(actVal: Val|null, label: string, container: HTMLElement, engineLayer: Module, filterPopup: HTMLElement|null): HTMLElement[] | null {
    if (!actVal || !actVal.data) return null;
    if (!filterPopup)   return null;

    const col = document.createElement('div');
    col.className = 'layer-column';
    const currentLayerElements: HTMLElement[] = [];    // elements to draw lines TO
    
    const shape = actVal.shape;
    const isSpatial = shape.length === 4;

    if (isSpatial) {    // conv and maxpool
        const [_, H, W, C] = shape;
        for (let m=0; m<C; m++) {
            const canv = document.createElement('canvas');
            const mapSize = H*W;
            const mapData = new Float64Array(mapSize);
            for (let pix=0; pix<mapSize; pix++) {
                mapData[pix] = actVal.data[pix*C + m];
            }
            renderFeatureMap(canv, mapData, W, H, 1);
            const maxDisplayDim = 48;
            let displayW, displayH;
            if (W >= H) {       // Wider or square
                displayW = maxDisplayDim;
                displayH = Math.round(maxDisplayDim * (H / W)) || 1;
            } else {            // Taller
                displayH = maxDisplayDim;
                displayW = Math.round(maxDisplayDim * (W / H)) || 1;
            }
            canv.style.width = `${displayW}px`
            canv.style.height = `${displayH}px`
            canv.title = `Channel ${m + 1}/${C} (Size: ${W}x${H})`;

            col.appendChild(canv);
            currentLayerElements.push(canv);
        }
    } else {    // dense and flatten
        const numNeurons = shape[1] || 1;
        const canv = document.createElement('canvas');
        renderFeatureMap(canv, actVal.data, 1, numNeurons, 1);
        
        canv.style.width = '16px';
        canv.style.height = '512px';
        canv.title = `${numNeurons} neurons`;

        col.appendChild(canv);
        currentLayerElements.push(canv);
    }

    const layerLabel = document.createElement('div');
    layerLabel.className = 'layer-label';
    layerLabel.innerText = label;
    col.appendChild(layerLabel);
    container.appendChild(col);

    // adding hover listener for conv layers
    if (engineLayer instanceof Conv && filterPopup) {
        col.addEventListener('mouseenter', ()=> {drawConvFilters(filterPopup, engineLayer); filterPopup.style.display = 'flex';});
        col.addEventListener('mousemove', (e)=> {
            filterPopup.style.left = `${e.pageX + 15}px`; 
            filterPopup.style.top = `${e.pageY + 30}px`;});
        col.addEventListener('mouseleave', ()=> {filterPopup.style.display = 'none';});
    }

    return currentLayerElements;
}

export function renderNetworkGraph(container: HTMLElement, actData: LayerOutputData[], model: Sequential, sampleX: Val, sampleY_label: number) {
    container.innerHTML = '';
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id = 'graph-svg';
    container.appendChild(svg);

    const filterPopup = document.getElementById('filter-popup');

    // render input img
    const inputCol = document.createElement('div');
    inputCol.className = 'layer-column';
    const inputCanv = document.createElement('canvas');
    inputCanv.className = 'input-image-canvas';
    const [_, H_in, W_in, C_in] = sampleX.shape;
    renderFeatureMap(inputCanv, sampleX.data, W_in, H_in, C_in);
    inputCanv.style.width = '64px';
    inputCanv.style.height = '64px';
    const inputLabel = document.createElement('div');

    const labelDisplayDiv = document.createElement('div');
    labelDisplayDiv.className = 'label-display';
    const trueLabelDiv = document.createElement('div');
    trueLabelDiv.innerHTML = `True Label: <span>${sampleY_label}</span>`;
    const predLabelDiv = document.createElement('div');
    predLabelDiv.innerHTML = `Predicted: <span>-</span>`;
    labelDisplayDiv.appendChild(trueLabelDiv);
    labelDisplayDiv.appendChild(predLabelDiv);

    inputLabel.className = 'layer-label';
    inputLabel.innerText = `Input Image\n${H_in}x${W_in}x${C_in}`;

    inputCol.appendChild(labelDisplayDiv);
    inputCol.appendChild(inputCanv);
    inputCol.appendChild(inputLabel);
    container.appendChild(inputCol);

    let prevLayerElements: HTMLElement[] = [inputCanv]; // elements to draw lines FROM
    let prevActVal: Val | null = sampleX;

    // render each layer's activations
    actData.forEach((layerOutput, i) => {
        const engineLayer = model.layers[i];

        // Create labels for Z and A
        let zLabel = `${engineLayer.constructor.name.replace('Layer', '')}\n(Z)`;
        let aLabel = `${engineLayer.constructor.name.replace('Layer', '')}\n(A)`;
        if (engineLayer instanceof MaxPool2D) zLabel = `MaxPool\n${engineLayer.pool_size}x${engineLayer.pool_size} P, S${engineLayer.stride}`;
        if (engineLayer instanceof Flatten) zLabel = "Flatten";
        if (!(engineLayer instanceof Dense || engineLayer instanceof Conv) || !engineLayer.activation) aLabel = '';

        // Render Z (pre-activation) and A (post-activation) columns
        const zElements = renderActivationCol(layerOutput.Z, zLabel, container, engineLayer, filterPopup);
        if(zElements) {
            drawConnectingLines(svg, prevLayerElements, prevActVal, zElements, engineLayer);
            prevLayerElements = zElements;
            prevActVal = layerOutput.Z;
        }

        const aElements = (layerOutput.A !== layerOutput.Z) ? renderActivationCol(layerOutput.A, aLabel, container, engineLayer, filterPopup) : null;
            if(aElements) {
            prevLayerElements = aElements;  // No lines between Z and A, just update the source for next layer
            prevActVal = layerOutput.A;
        }
    });

    const finalAct = actData[actData.length-1]?.A;
    if (!finalAct || !finalAct.data) return;
    
    const prob = finalAct.data;
    let maxProb=-1;
    let predictedLabel=-1;
    for (let j=0; j<prob.length; j++) {
        if (prob[j]>maxProb) {
            maxProb=prob[j];
            predictedLabel=j;
        }
    }
    const predSpan = predLabelDiv.querySelector('span');
    if (!predSpan) return;
    predSpan.textContent = predictedLabel.toString();
    if (predictedLabel === sampleY_label) {
        predSpan.style.color = '#28a745'; // Green
        predSpan.style.fontWeight = 'bold';
    } else {
        predSpan.style.color = '#dc3545'; // Red
        predSpan.style.fontWeight = 'bold';
    }
}

function drawConnectingLines(svg: SVGSVGElement, fromElements: HTMLElement[], fromActVal: Val|null, toElements: HTMLElement[], toLayer: Module): void {
    if (!fromActVal)    return;
    const containerRect = svg.parentElement!.getBoundingClientRect();

    if (toLayer instanceof Dense && toElements.length === 1) {
        const nin = toLayer.nin;
        const nout = toLayer.nout;
        const weights = toLayer.W.data;

        let maxAbsWeight = 0;
        for (const w of weights) {
            const absW = Math.abs(w);
            if (absW > maxAbsWeight) maxAbsWeight = absW;
        }
        if (maxAbsWeight === 0) maxAbsWeight = 1;

        const toEl = toElements[0];
        const toRect = toEl.getBoundingClientRect();
        const toX = toRect.left - containerRect.left;

        const fromShape = fromActVal.shape;
        const isFromSpatial = fromShape.length === 4;

        const maxLinesToDraw = 150;
        for (let i=0; i<maxLinesToDraw; i++) {
            const in_neuron_idx = Math.floor(Math.random()*nin);
            const out_neuron_idx = Math.floor(Math.random()*nout);

            const toY = (toRect.top - containerRect.top) + (toRect.height*(out_neuron_idx+0.5)/nout);

            let fromX: number;
            let fromY: number;

            if (isFromSpatial) {
                const [_, H, W, C] = fromShape;
                const mapSize = H*W;
                const c_idx = Math.floor(in_neuron_idx/mapSize);
                const pixel_in_map = in_neuron_idx%mapSize;
                const h_idx = Math.floor(pixel_in_map/W);

                const fromEl = fromElements[c_idx];
                if (!fromEl) continue;
                const fromRect = fromEl.getBoundingClientRect();
                fromX = fromRect.right - containerRect.left;
                fromY = (fromRect.top-containerRect.top) + (fromRect.height*(h_idx+0.5)/H);
            } else {
                const fromEl = fromElements[0];
                const fromRect = fromEl.getBoundingClientRect();
                fromX = fromRect.right - containerRect.left;
                fromY = (fromRect.top - containerRect.top) + (fromRect.height*(in_neuron_idx+0.5)/nin);
            }

            // flat index is in_neuron_idx * nout + out_neuron_idx for weight [nin, nout]
            const weight_idx = in_neuron_idx*nout + out_neuron_idx;
            const weight = weights[weight_idx];
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', fromX.toString());
            line.setAttribute('y1', fromY.toString());
            line.setAttribute('x2', toX.toString());
            line.setAttribute('y2', toY.toString());

            const weightMagNorm = Math.abs(weight)/maxAbsWeight;
            const opacity = Math.min(0.7, weightMagNorm*1.5);
            const thickness = Math.min(2.5, 0.1+weightMagNorm*4);
            line.setAttribute('stroke', weight>0?'rgba(0, 100, 255, 0.6)': 'rgba(255, 50, 0, 0.6)');
            line.setAttribute('stroke-opacity', opacity.toString());
            line.setAttribute('stroke-width', thickness.toString());
            svg.appendChild(line);
        }
    } else {        // for conv/pool
        toElements.slice(0, 50).forEach(toEl=> {
            fromElements.slice(0, 50).forEach(fromEl => {
                const fromRect = fromEl.getBoundingClientRect();
                const toRect = toEl.getBoundingClientRect();
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', (fromRect.right - containerRect.left).toString());
                line.setAttribute('y1', (fromRect.top + fromRect.height/2 - containerRect.top).toString());
                line.setAttribute('x2', (toRect.left - containerRect.left).toString());
                line.setAttribute('y2', (toRect.top + toRect.height/2 - containerRect.top).toString());
                line.setAttribute('stroke-width', '0.5');
                svg.appendChild(line);
            });
        });
    }
}

// renders conv filters which show up on hover into a popup element
function drawConvFilters(popupEl: HTMLElement, convLayer: Conv): void {
    popupEl.innerHTML = '';
    const filters = convLayer.kernel;
    const [C_out, K, _, C_in] = filters.shape;

    for (let c_out=0; c_out<C_out; c_out++) {
        const filterData = new Float64Array(K*K);
        for (let i=0; i<K*K; i++) {
            const h=Math.floor(i/K);
            const w=i%K;
            const flatIdx = c_out*(K*K*C_in) + h*(K*C_in) + w*C_in + 0;
            filterData[i] = filters.data[flatIdx];
        }
        const canv = document.createElement('canvas');
        renderFeatureMap(canv, filterData, K, K, 1);
        canv.style.width = '32px';
        canv.style.height = '32px';
        popupEl.appendChild(canv);
    }
}