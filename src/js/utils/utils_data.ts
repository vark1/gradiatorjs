export function imgToVector(img: HTMLCanvasElement) {
    // get image data
    var canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext('2d')
    if(!ctx)    return;
    ctx.drawImage(img, 0, 0);
    var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return Float64Array.from(img_data.data);
}

