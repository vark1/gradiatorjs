import { Conv, Dense, Flatten, MaxPool2D, Module, Sequential } from "../nn/nn.js";
import { calculateMinMax } from "../utils/utils.js";
import { Val } from "../Val/val.js";
import { LayerOutputData } from "../types_and_interfaces/vis_interfaces.js";

function renderFeatureMap(canvas: HTMLCanvasElement, mapData: Float64Array, W: number, H: number) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = W;
    canvas.height = H;

    const mm = calculateMinMax(mapData);
    const imageData = ctx.createImageData(W, H);
    const dataArr = imageData.data;

    for (let i=0; i<mapData.length; i++) {
        const normalized = mm.dv===0?0.5:(mapData[i]-mm.minv)/mm.dv;
        const grayVal = Math.floor(normalized*255);
        const p = i*4;
        dataArr[p] = dataArr[p+1] = dataArr[p+2] = grayVal;
        dataArr[p+3] = 255;
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
            renderFeatureMap(canv, mapData, W, H);
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
        renderFeatureMap(canv, actVal.data, 1, numNeurons);
        
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

export function renderNetworkGraph(container: HTMLElement, actData: LayerOutputData[], model: Sequential, sampleX: Val) {
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
    renderFeatureMap(inputCanv, sampleX.data, W_in, H_in);
    inputCanv.style.width = '64px';
    inputCanv.style.height = '64px';
    const inputLabel = document.createElement('div');
    inputLabel.className = 'layer-label';
    inputLabel.innerText = `Input Image\n${H_in}x${W_in}x${C_in}`;
    inputCol.appendChild(inputCanv);
    inputCol.appendChild(inputLabel);
    container.appendChild(inputCol);

    let prevLayerElements: HTMLElement[] = [inputCanv]; // elements to draw lines FROM

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
            drawConnectingLines(svg, prevLayerElements, zElements, engineLayer);
            prevLayerElements = zElements;
        }

        const aElements = (layerOutput.A !== layerOutput.Z) ? renderActivationCol(layerOutput.A, aLabel, container, engineLayer, filterPopup) : null;
            if(aElements) {
            prevLayerElements = aElements;  // No lines between Z and A, just update the source for next layer
        }
    });
}

function drawConnectingLines(svg: SVGSVGElement, fromElements: HTMLElement[], toElements: HTMLElement[], toLayer: Module): void {
    const containerRect = svg.parentElement!.getBoundingClientRect();

    let avgWeightMag = 0.1;
    if (toLayer instanceof Dense && toLayer.W) {
        const weights = toLayer.W.data;
        let sumMag = 0;
        for(const w of weights) {
            sumMag+=Math.abs(w);
        }
        avgWeightMag = sumMag/weights.length;
    }

    const maxLines = 50;
    let linesDrawn = 0;

    for (const toEl of toElements) {
        for (const fromEl of fromElements) {
            if (linesDrawn >= maxLines) break;

            const fromRect = fromEl.getBoundingClientRect();
            const toRect = toEl.getBoundingClientRect();

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', (fromRect.right - containerRect.left).toString());
            line.setAttribute('y1', (fromRect.top + fromRect.height/2 - containerRect.top).toString());
            line.setAttribute('x2', (toRect.left - containerRect.left).toString());
            line.setAttribute('y2', (toRect.top + toRect.height/2 - containerRect.top).toString());

            const thickness = (toLayer instanceof Dense) ? Math.min(3, 0.1 + avgWeightMag * 20) : 0.5;
            line.setAttribute('stroke-width', thickness.toString());
            svg.appendChild(line);
            linesDrawn++;
        }
        if (linesDrawn >= maxLines) break;
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
        renderFeatureMap(canv, filterData, K, K);
        canv.style.width = '32px';
        canv.style.height = '32px';
        popupEl.appendChild(canv);
    }
}