import {NDArray} from '../types'


function imgToNd(img) {

    let canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height

    let ctx = canvas.getContext('2d')

    ctx.drawImage(img, 0, 0)
    let img_data = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let p = img_data.data
    let w = img.width
    let h = img.height
    let pv = []
    for (let i=0; i<p.length; i++) {
        pv.push(p[i]/255.0-0.5)
    }

    let x : NDArray = []
}

function imgToVector(img) {
    // imgToNd
    // flatten ND
    // reshape ND
    // transpose
    // resultant nd
}