"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.imgToVector = imgToVector;
exports.loadData = loadData;
exports.load_dataset = load_dataset;

let DATASET_HDF5 = null;
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
        if (!barr) {
            console.error('Failed to read file');
            return;
        }
        DATASET_HDF5 = new window.hdf5.File(barr, datafilename);
        console.log(DATASET_HDF5)
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

function load_dataset(){

    let train_dataset = DATASET_HDF5
    let train_set_x_orig = train_dataset.get('train_set_x').value
    let train_set_y_orig = train_dataset.get('train_set_y').value

    let test_dataset = DATASET_HDF5
    let test_set_x_orig = test_dataset.get('test_set_x').value
    let test_set_y_orig = test_dataset.get('test_set_y').value

    let classes = test_dataset.get("list_classes") // the list of classes
    
    train_set_y_orig = reshape(train_set_y_orig, [1, train_set_y_orig.shape[0]])
    test_set_y_orig = reshape(test_set_y_orig, [1, test_set_y_orig.shape[0]])

    return [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]
}

