export let DATASET_HDF5_TRAIN = null;
export let DATASET_HDF5_TEST = null;

export function imgToVector(img: HTMLCanvasElement) {
    // get image data
    var canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext('2d')
    if(!ctx)    return;
    ctx.drawImage(img, 0, 0);
    var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var p = img_data.data; // A flat array of RGBA values
    return Float64Array.from(img_data.data);
}

export function loadData() {
    var file_input = <HTMLInputElement>document.getElementById('datafile');
    if (!file_input.files) return;
    var file = file_input.files[0]; // only one file allowed
    var datafilename = file.name;
    var reader = new FileReader();
    reader.onloadend = function (evt) {
        var barr = evt.target!.result;
        if (!barr) {
            console.error('Failed to read file');
            return;
        }
        let dataset = new hdf5.File(barr, datafilename);
        if(datafilename.includes("train")) {
            DATASET_HDF5_TRAIN = dataset
        } else if (datafilename.includes("test")) {
            DATASET_HDF5_TEST = dataset
        } else {
            console.log("Train or Test not found in dataset filename, maybe wrong dataset?")
        }
        console.log("train", DATASET_HDF5_TRAIN)
        console.log("test", DATASET_HDF5_TEST)

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