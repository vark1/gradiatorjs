"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.imgToVector = imgToVector;
exports.loadData = loadData;
function imgToVector(img) {
    // get image data
    var canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var p = img_data.data; // A flat array of RGBA values
    var pv = [];
    for (var i = 0; i < p.length; i++) { // normalizing these values from -0.5 to 0.5
        pv.push(p[i] / 255.0 - 0.5);
    }
    ctx.putImageData(img_data, 0, 0);
    return pv;
}

function loadData() {
    var file_input = document.getElementById('datafile');
    var file = file_input.files[0]; // only one file allowed
    var datafilename = file.name;
    var reader = new FileReader();
    reader.onloadend = function (evt) {
        var barr = evt.target.result;
        var f = new window.hdf5.File(barr, datafilename);
        console.log(f);
        // do something with f...
    };
    reader.readAsArrayBuffer(file);
    file_input.value = "";
}

document.addEventListener('DOMContentLoaded', function () {
    var fileInput = document.getElementById('datafile');
    if (fileInput) {
        fileInput.addEventListener('change', loadData);
    }
});