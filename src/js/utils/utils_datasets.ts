import { Val } from "../Val/val.js";
import * as ops from '../Val/ops.js'

export let DATASET_HDF5_TRAIN = null;
export let DATASET_HDF5_TEST = null;

export function catvnoncat_loadData() {
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

export function catvnoncat_prepareDataset() {

    // shape of dataset: (m, 64, 64, 3)
    let train_set_x = (<any>DATASET_HDF5_TRAIN).get('train_set_x')
    let train_x_og = new Val(train_set_x.shape)
    train_x_og.data = Float64Array.from(train_set_x.value)

    let train_set_y = (<any>DATASET_HDF5_TRAIN).get('train_set_y')    
    let train_y_og = new Val(train_set_y.shape)
    train_y_og.data = Float64Array.from(train_set_y.value)

    let test_set_x = (<any>DATASET_HDF5_TEST).get('test_set_x')
    let test_x_og = new Val(test_set_x.shape)
    test_x_og.data = Float64Array.from(test_set_x.value)

    let test_set_y = (<any>DATASET_HDF5_TEST).get('test_set_y')
    let test_y_og = new Val(test_set_y.shape)
    test_y_og.data = Float64Array.from(test_set_y.value)

    let classes = (<any>DATASET_HDF5_TEST).get("list_classes")
    train_y_og = train_y_og.reshape([train_y_og.shape[0], 1])  // [m, nout(1 here)]
    test_y_og = test_y_og.reshape([test_y_og.shape[0], 1])

    console.log(`# Training examples: ${train_x_og.shape[0]}`)
    console.log(`# Testing examples: ${test_x_og.shape[0]}`)
    
    let train_x_flatten = train_x_og.reshape([train_x_og.shape[0], train_x_og.size/train_x_og.shape[0]])  // [m, nin]
    let test_x_flatten = test_x_og.reshape([test_x_og.shape[0], test_x_og.size/test_x_og.shape[0]])
    
    let train_x = ops.div(train_x_flatten, 255)
    let test_x = ops.div(test_x_flatten, 255)

    const nin = train_x.shape[1];   // train_x.shape = [m, nin]
    const inH = train_set_x.shape[1];
    const inW = train_set_x.shape[2];

    return {
        "train_x_og": train_x_og,
        "train_x_flatten": train_x,
        "train_y": train_y_og,
        "test_x_og": test_x_og,
        "test_x_flatten": test_x_flatten,
        "test_y": test_y_og,
        "nin": nin,
        "inh": inH,
        "inW": inW
    }
}

document.addEventListener('DOMContentLoaded', function () {
    var fileInput = document.getElementById('datafile');
    if (fileInput) {
        fileInput.addEventListener('change', catvnoncat_loadData);
    }
});