export function imgToVector(img) {

    // get image data
    let canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height
    
    const ctx = canvas.getContext('2d')

    ctx.drawImage(img, 0, 0)
    let img_data = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let p = img_data.data   // A flat array of RGBA values
    let pv = []
    for (let i=0; i<p.length; i++) {    // normalizing these values from -0.5 to 0.5
        pv.push(p[i]/255.0-0.5)
    }
    
    ctx.putImageData(img_data, 0, 0);
    return pv
}