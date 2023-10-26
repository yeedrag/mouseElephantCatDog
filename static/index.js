import { getParameters } from "./parameterBarHandler.js";
//I don't know how to avoid using all the functions in global scope. all items with "window." prefix should be fixed in some way.

//initailize and util------------------------------------------------------------------------------------------------------------------

var idCur = -1; //the id of the last added block
var backendOutput = []; //the output of backend, it should be an array of dictionary
const activations = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']; //set activations name

window.URL.createObjectURL = function() {};

import blockParameterData from './test.json' assert { type: 'json' };;
function getIdNumber(string) {//get id number from any form of id string whose id number is its suffix
    var idNum=0;
    for(let i = string.length-1, k = 1;string[i] != ':' ; i-- , k*=10) {
        idNum+=k*parseInt(string[i]);
    }
    return idNum;
}
function test() { //just to test whether a event is triggered correctly
    console.log("test");
}
window.test = test;

//initailize and util-----------------------------------------------------------------------------------------------------------------------------

//drag and drop--------------------------------------------------------------------------------------------------------------------------

function drag_start(event) {
    var style = window.getComputedStyle(event.target, null);
    var idNum = getIdNumber(event.target.id);
    event.dataTransfer.setData("text/plain",
    (parseInt(style.getPropertyValue("left"),10) - event.clientX) + ',' + (parseInt(style.getPropertyValue("top"),10) - event.clientY)+','+idNum);
    return false;
} 
window.drag_start = drag_start;

function drag_over(event) { 
    event.preventDefault(); 
    return false; 
} 
window.drag_over = drag_over;

function drop(event) { 
    var offset = event.dataTransfer.getData("text/plain").split(',');
    var dm = document.getElementById('blockID:'+offset[2]);
    dm.style.left = (event.clientX + parseInt(offset[0],10)) + 'px';
    dm.style.top = (event.clientY + parseInt(offset[1],10)) + 'px';
    event.preventDefault();
    return false;
} 
window.drop = drop;

//drag and drop--------------------------------------------------------------------------------------------------------------------------

//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------

function openParameterBar(event) {
    var idNum = getIdNumber(event.target.id);
    
        for(let i = 0; i <= idCur; i++) {
            let curBar = document.getElementById("PBarID:" + i);
            if(curBar.style.width == "250px" && i != idNum) {
                curBar.style.width = "0px";
            }
        }
        document.getElementById("PBarID:" + idNum).style.width = "250px";
    
    //document.getElementById("main").style.marginLeft = "250px";
}
window.openParameterBar = openParameterBar;

function closeParameterBar(event) {
    var idNum = getIdNumber(event.target.id);
    document.getElementById("PBarID:" + idNum).style.width = "0";
    //document.getElementById("main").style.marginLeft= "0";
}
window.closeParameterBar = closeParameterBar;

//dbclick for side collapse bar------------------------------------------------------------------------------------------------------------

//add a new block and parameter bar----------------------------------------------------------------------------------------------------------------------------

function addBlock() {
    idCur++;
    const ele = document.createElement("div"); //create the block to add
    const curType = document.getElementById('addBlock-modal-selecter').value; // get block type
    ele.setAttribute('id', 'blockID:' + idCur); 
    ele.setAttribute('class', 'block');
    // set drag event 
    ele.setAttribute('draggable', true);
    ele.setAttribute('ondragstart', "drag_start(event)");

    ele.setAttribute('ondblclick','openParameterBar(event)')//set open parameterBar event
    ele.setAttribute('blockType', curType);//set type of the block 

    //set text on the block
    const textOfBlock = document.createElement("h2");
    textOfBlock.textContent = curType;
    ele.appendChild(textOfBlock);

    //set a delete button on the block
    const deleteBlockButton = document.createElement("a");
    deleteBlockButton.setAttribute('href', 'javascript:void(0)');
    deleteBlockButton.setAttribute('id', 'addBlockCloseButton:' + idCur);
    deleteBlockButton.setAttribute('onclick','deleteBlock(event)');
    deleteBlockButton.textContent = "X";
    ele.appendChild(deleteBlockButton);

    const workspace = document.getElementById('workSpace');
    workspace.appendChild(ele);

    //call function to add a new parameterBar for the new block
    addParameterBar(idCur, curType);
}
window.addBlock = addBlock;

//parameterBar

function addParameterBar(idCur, curType) {
    const newBar = document.createElement("div");
    newBar.setAttribute('class', 'parameterBar');
    newBar.setAttribute('id', 'PBarID:' + idCur);
    const paraCloseButton = document.createElement("a");
    paraCloseButton.setAttribute('href', 'javascript:void(0)');
    paraCloseButton.setAttribute('id', 'PBarClsID:' + idCur);
    paraCloseButton.setAttribute('onclick','closeParameterBar(event)');
    paraCloseButton.textContent = "X";
    newBar.appendChild(paraCloseButton);

    //set parameters of the new block in the new parameterBar
    let isActivation = activations.includes(curType);
    getParameters(newBar, blockParameterData, idCur, curType, backendOutput, isActivation);

    const workspace = document.getElementById('workSpace');
    workspace.appendChild(newBar);

}
window.addParameterBar = addParameterBar;

//add a new block and parameter bar----------------------------------------------------------------------------------------------------------------------------

//delete a block and paarmeter bar----------------------------------------------------------------------------------------------------------------------------

function deleteBlock(event) {
    var idNum = getIdNumber(event.target.id);
    //console.log(idNum);
    document.getElementById('blockID:' + idNum).remove();
    for(let i = idNum+1;i <= idCur; i++) {
        let curBlk = document.getElementById('blockID:' + i);
        curBlk.setAttribute('id', 'blockID:' + (i-1));
        let curClsB = document.getElementById('addBlockCloseButton:' + i);
        curClsB.setAttribute('id', 'addBlockCloseButton:' + (i-1));
    }

    backendOutput = backendOutput.slice(0, idNum).concat(backendOutput.slice(idNum+1));
    console.log(backendOutput);
    
    deleteParameterBar(idNum);
    idCur--;
    //console.log(idCur);
}
window.deleteBlock = deleteBlock;

function deleteParameterBar(idNum) {
    document.getElementById('PBarID:' + idNum).remove();
    for(let i = idNum+1;i <= idCur; i++) {
        let curPara = document.getElementById('PBarID:' + i);
        curPara.setAttribute('id', 'PBarID:' + (i-1));
    }
    //console.log(idCur);
}


//delete block----------------------------------------------------------------------------------------------------------------------------

//compile the model------------------------------------------------------------------------------------------------------------------------
function compileModel() {
    
    for(let idNum = 0; idNum <= idCur; idNum++) {
        var curBlkDict = {};
    }
}
window.compileModel = compileModel;

//input change event------------------------------------------------------------------------------------------------------------------------

function onInputChange(event) {
    var idNum = event.target.id.split(':')[1];
    var parameterName = event.target.id.split(':')[0];
    if(parameterName == 'child'||parameterName == 'parent') {
        backendOutput[idNum][parameterName] = event.target.value.split(',').map((x)=>parseInt(x));
    } else {
        if(event.target.type == 'text') {
            backendOutput[idNum]["args"][parameterName] = event.target.value.split(',').map((x)=>parseInt(x));
        }else {
            backendOutput[idNum]["args"][parameterName] = event.target.value;
        }
    }
    console.log(backendOutput);
}
window.onInputChange = onInputChange;

//input change event------------------------------------------------------------------------------------------------------------------------


//compile the model------------------------------------------------------------------------------------------------------------------------

function sendingData() {//sending frontend data to the backend
    fetch('/index', {
        headers : {'Content-Type' : 'application/json'},
        method : 'POST',
        body : JSON.stringify( {backendOutput})
    })
     
    
    //let el = document.createElement("a"); 
    // creates anchor element but doesn't add it to the DOM
    //el.href = "{{ url_for('download', filename='test.onnx')}}"; // set the href attribute attribute
    //el.click();
    //el.remove();
}
window.sendingData = sendingData;

function downloadFile(){

}
window.downloadFile = downloadFile;
